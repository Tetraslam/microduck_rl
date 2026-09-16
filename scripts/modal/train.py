"""Single-GPU training on Modal; see docs/modal.md."""

from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import modal

ROOT = (
    Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/microduck")
)
GPU = os.environ.get("MICRODUCK_MODAL_GPU", "B200")
VOLUME_NAME = "microduck-runs"
app = modal.App("microduck-rl")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Install the lockfile in a cached layer before adding frequently edited source.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libegl1", "libgl1", "ffmpeg")
    .env({"UV_HTTP_TIMEOUT": "600", "MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1"})
    .uv_sync(str(ROOT), frozen=True, uv_version="0.11.21")
    .add_local_dir(
        ROOT / "src",
        "/opt/microduck/src",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(ROOT / "pyproject.toml", "/opt/microduck/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/opt/microduck/uv.lock", copy=True)
    .add_local_file(ROOT / "README.md", "/opt/microduck/README.md", copy=True)
    .add_local_file(ROOT / "LICENSE", "/opt/microduck/LICENSE", copy=True)
    .run_commands(
        "/.uv/uv pip install --python /.uv/.venv/bin/python --no-deps /opt/microduck"
    )
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    volumes={"/runs": volume},
    timeout=24 * 60 * 60,
    retries=0,
)
def train_remote(options: dict) -> str:
    from mjlab_microduck.modal_worker import train

    return train(
        root=Path("/runs"),
        source=Path("/opt/microduck"),
        commit=volume.commit,
        **options,
    )


@app.local_entrypoint()
def main(
    task: str = "Mjlab-Roulade-Flat-MicroDuck",
    num_envs: int = 4096,
    iterations: int = 1000,
    save_interval: int = 50,
    seed: int = 42,
    name: str = "duck",
    resume_from: str = "",
    train_args: str = "",
    wait: bool = False,
):
    """Iterations are additional updates, including when resuming a checkpoint."""
    if not name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in name
    ):
        raise ValueError("Use letters, numbers, hyphens or underscores for --name")
    if not wait and not any(flag in sys.argv for flag in ("--detach", "-d")):
        raise ValueError("Use modal run --detach for background jobs, or pass --wait")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{name}-{uuid.uuid4().hex[:8]}"
    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "requested_gpu": GPU,
    }
    print(f"Run: {run_id}\nVolume: {VOLUME_NAME}\nGPU: {GPU}", flush=True)
    options = dict(
        run_id=run_id,
        task=task,
        num_envs=num_envs,
        iterations=iterations,
        save_interval=save_interval,
        seed=seed,
        extra_args=shlex.split(train_args),
        resume_from=resume_from,
        provenance=provenance,
    )
    call = train_remote.spawn(options)
    print(f"Submitted function call: {call.object_id}", flush=True)
    if wait:
        call.get()
        print(f"Saved to {VOLUME_NAME}/{run_id}", flush=True)
    print(
        f"mkdir -p logs/modal\nmodal volume get {VOLUME_NAME} {run_id} logs/modal/",
        flush=True,
    )
