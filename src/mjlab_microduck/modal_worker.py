"""Modal worker: native mjlab training with explicitly committed checkpoints.

Each invocation owns a new run directory. Checkpoints are written to a temporary
name, atomically renamed after torch.save closes them, then committed to the
Volume. A killed writer can leave a temporary file, never a partial model_*.pt.
Resume restores learning/optimizer/normalizer and curriculum state, not the
in-flight simulator or rollout buffer. It always writes into a new run.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Callable


def atomic_checkpoint(
    path: str, write: Callable[[str], None], commit: Callable[[], None]
):
    """Only expose and commit a checkpoint after its writer has returned."""
    destination = Path(path)
    temporary = destination.with_suffix(".pt.incomplete")
    try:
        write(str(temporary))
        temporary.replace(destination)
        commit()
    finally:
        temporary.unlink(missing_ok=True)


def resolve_checkpoint(root: Path, relative: str) -> Path:
    checkpoint = (root / relative).resolve()
    if not checkpoint.is_relative_to(root.resolve()):
        raise ValueError("Resume checkpoint must be inside the runs volume")
    if not re.fullmatch(r"model_\d+\.pt", checkpoint.name):
        raise ValueError("Resume checkpoint must be named model_<iteration>.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def train(
    *,
    root: Path,
    source: Path,
    run_id: str,
    task: str,
    num_envs: int,
    iterations: int,
    save_interval: int,
    seed: int,
    extra_args: list[str],
    resume_from: str,
    provenance: dict,
    commit: Callable[[], None],
) -> str:
    import mjlab.tasks  # noqa: F401, registers plugins
    import torch
    import tyro
    from mjlab.scripts import train as training
    from mjlab_microduck.export import ExportConfig, run_export

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Invalid run id")
    if min(num_envs, iterations, save_interval) < 1:
        raise ValueError("num_envs, iterations and save_interval must be positive")
    checkpoint = resolve_checkpoint(root, resume_from) if resume_from else None
    if checkpoint:
        parent_metadata = checkpoint.parent / "run.json"
        if parent_metadata.exists():
            parent = json.loads(parent_metadata.read_text())
            if parent["task"] != task:
                raise ValueError(
                    "Resume is same-task only; cross-task warm starts need a separate recipe"
                )

    # Extra args are native task/reward/algorithm options. Lifecycle settings below
    # stay explicit so the run metadata and actual execution cannot disagree.
    cfg = tyro.cli(
        training.TrainConfig,
        args=extra_args,
        default=training.TrainConfig.from_task(task),
        config=mjlab.TYRO_FLAGS,
    )
    cfg = replace(cfg, wandb_run_path=None, wandb_checkpoint_name=None, gpu_ids=[0])
    cfg.env.scene.num_envs = num_envs
    cfg.agent.max_iterations = iterations
    cfg.agent.save_interval = save_interval
    cfg.agent.seed = seed
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    cfg.agent.run_name = run_id
    cfg.agent.resume = checkpoint is not None
    if checkpoint:
        cfg.agent.load_checkpoint = re.escape(checkpoint.name) + "$"
        # get_checkpoint_path accepts load_run as a regex matched against folders
        # under log_dir.parent. Imports live in that same volume root.
        cfg.agent.load_run = re.escape(checkpoint.parent.name) + "$"
        if checkpoint.parent.parent != root.resolve():
            raise ValueError(
                "Resume path must have shape <run-id>/model_<iteration>.pt"
            )

    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        provenance,
        task=task,
        run_id=run_id,
        num_envs=num_envs,
        additional_iterations=iterations,
        save_interval=save_interval,
        seed=seed,
        extra_args=extra_args,
        resume_from=resume_from,
        gpu=torch.cuda.get_device_name(0),
        state="running",
    )
    metadata_path = run_dir / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with tarfile.open(run_dir / "source.tar.gz", "w:gz") as archive:
        for name in ("src", "pyproject.toml", "uv.lock", "README.md", "LICENSE"):
            archive.add(source / name, arcname=name)
    commit()

    original_loader = training.load_runner_cls
    runner_cls = original_loader(task) or training.MjlabOnPolicyRunner

    class PersistentRunner(runner_cls):
        def save(self, path, infos=None):
            original_save = super().save
            atomic_checkpoint(path, lambda tmp: original_save(tmp, infos), commit)
            print(
                f"[modal] Durable checkpoint: {Path(path).relative_to(root)}",
                flush=True,
            )

    training.load_runner_cls = lambda _: PersistentRunner
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.pop("MICRODUCK_WARM_START", None)
    try:
        training.run_train(task, cfg, run_dir)
        latest = max(
            run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])
        )
        run_export(
            task,
            ExportConfig(
                checkpoint_file=str(latest),
                num_envs=1,
                onnx_file=str(run_dir / "policy.onnx"),
            ),
        )
        metadata["state"] = "completed"
        metadata["checkpoint"] = latest.name
    except BaseException as error:
        metadata["state"] = "failed"
        metadata["error"] = repr(error)
        raise
    finally:
        training.load_runner_cls = original_loader
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        commit()
    return str(run_dir.relative_to(root))
