# Modal training

Run GPU training remotely, keep viewers on the laptop. Uses the active Modal
profile/workspace (`modal profile list`), the repository's frozen `uv.lock`, and
a snapshot of the current source. Local code changes are included without a push.
The Modal CLI is separate from the training environment; tested with 1.5.2.

## Start a run

From the repository root, first smoke-test the task:

```bash
modal run scripts/modal/train.py --wait --task Mjlab-Roulade-Flat-MicroDuck \
  --num-envs 64 --iterations 5 --name smoke
```

Then launch detached so closing the terminal or rebooting the laptop cannot
stop training. Submission returns immediately with a function-call ID:

```bash
modal run --detach scripts/modal/train.py \
  --task Mjlab-Roulade-Flat-MicroDuck --num-envs 4096 \
  --iterations 10000 --name somersault
```

The task's original rewards, curriculum, normalizers and PPO settings are used.
Defaults: 4096 environments, 1000 additional iterations, seed 42, a checkpoint
every 50 iterations, TensorBoard logging. Each job gets one GPU, four CPU cores,
16 GiB host memory, and a 24-hour timeout. GPU type is selected with
`MICRODUCK_MODAL_GPU` (default B200). No cloud services stay deployed after the job finishes.

`--train-args` forwards quoted native mjlab configuration options, for example:

```bash
modal run --detach scripts/modal/train.py --name exploration \
  --train-args '--agent.algorithm.entropy-coef 0.02'
```

Task, seed, environment count, iteration budget, save interval, local TensorBoard
logging, and resume source are controlled by this launcher's explicit options.
Use `uv run list-envs` to choose another task. Smoke-test each new task/config
before a long run.

## Watch, stop, resume

The launch prints a Modal dashboard URL and a unique run ID. Logs are available
there or with `modal app logs <app-id>`. Metrics are stored as TensorBoard events
alongside the checkpoints.

```bash
modal volume ls microduck-runs
modal volume ls microduck-runs <run-id>
modal app stop <app-id>

# Continue a specific checkpoint, writing into a NEW run directory.
modal run --detach scripts/modal/train.py --name continued \
  --resume-from <run-id>/model_500.pt --iterations 2000
```

`--iterations` means **additional PPO updates**, not a target final iteration.
The pinned runner restores its saved iteration number and repeats that label on
the first resumed update. Optimizer, normalizer and curriculum step counter are
restored. Simulator/episode state and partially collected rollouts are not.
Resume uses the current source and task configuration: keep the same task and
recipe unless intentionally changing an experiment. Cross-task warm starts are
not provided by this launcher.

Stopping loses progress since the most recent committed checkpoint. There is no
free cloud equivalent of keeping a local process frozen; stop and resume instead.
Jobs have no automatic retries, so a preemption or timeout requires an explicit
resume from a saved checkpoint. A forcibly killed run can retain `state: running`
in its `run.json`; the Modal app status is authoritative for liveness.

## Download and play

```bash
mkdir -p logs/modal
modal volume get microduck-runs <run-id> logs/modal/
uv run play Mjlab-Roulade-Flat-MicroDuck \
  --checkpoint-file logs/modal/<run-id>/model_9999.pt --num-envs 1
uv run tensorboard --logdir logs/modal
```

The local destination must be an existing directory (`logs/modal/`, not a new
run-specific path); Modal nests the run beneath it. Add `--force` to refresh a
previously downloaded run.

The completed run also contains `policy.onnx`, exported with the repository's
normalizer-preserving exporter. For a checkpoint downloaded before completion:

```bash
uv run scripts/export.py Mjlab-Roulade-Flat-MicroDuck \
  --checkpoint-file logs/modal/<run-id>/model_500.pt --num-envs 1 \
  --onnx-file logs/modal/<run-id>/policy.onnx
```

To continue a local checkpoint on Modal:

```bash
modal volume put microduck-runs path/to/model_9999.pt imported-roll/model_9999.pt
modal run --detach scripts/modal/train.py \
  --resume-from imported-roll/model_9999.pt --iterations 1000
```

Use a unique import directory, and the checkpoint's original task. Imports
without `run.json` cannot have their task identity checked automatically.

## Persistence contract

- A Modal invocation owns one unique `/runs/<run-id>` directory in the
  `microduck-runs` Volume. Concurrent experiments write separate directories.
- Native mjlab saves the checkpoint to `.pt.incomplete`; the wrapper renames it
  to `model_N.pt` only after the writer returns, then explicitly commits the
  Volume. Resume/download selects only complete checkpoint names.
- A failed write preserves any previous complete checkpoint. A failed commit
  fails the job rather than claiming the checkpoint is durable. Background Volume
  snapshots may contain incomplete files; those are never resume candidates.
- `run.json`, native config YAMLs, and `source.tar.gz` record what ran. The source
  archive contains the training package, robot assets, lockfile and project
  metadata. Source/metadata and final exports are explicitly committed too.
- Unit tests cover interrupted writes, commit failure and path validation. Real
  GPU smoke tests and cross-container resumes verify the native checkpoint
  payload, normalizer and curriculum lifecycle.

## GPU choice

Benchmark the actual task, not advertised tensor FLOPS. MuJoCo Warp physics and
contact processing dominate this small-policy workload. Compare cards at the
same environment count first; increasing the count changes PPO batch size and
samples per curriculum stage, so throughput alone does not prove faster learning.

L40S, A100-80GB, H100 and B200 work with the pinned CUDA 12 stack. Modal's B300
requires CUDA 13.1+, so it needs a separately validated dependency upgrade.
Use `H100!` for reproducible H100 benchmarking: `H100` may be upgraded to H200.

### Measured on 2026-09-16

Somersault task, resumed from the laptop's `model_9999.pt`, seed 42, 24 rollout
steps, four CPU cores, 16 GiB host memory. Each measurement is 40 actual PPO
updates; medians below exclude the first 10. Compilation, scheduling, Volume
commits and final export are outside the reported training-loop timing.
These are single short runs, not time-to-convergence comparisons.

| GPU | 4,096 envs (steps/s) | 16,384 envs (steps/s) |
|---|---:|---:|
| A100 SXM4 80 GB | 41,346 | Not tested |
| L40S | 53,735 | 66,145 |
| H100 SXM 80 GB | 56,705 | 86,945 |
| B200 | **84,534** | **127,840** |

**Default: B200 with 4,096 environments.** It was 1.49× faster than H100 and
1.57× faster than L40S at the established batch size. B200 with 16,384 envs
processes more samples per second, but an update takes about 3.08 s instead of
1.16 s and sees four times as many samples. Retuning batch size/curriculum for
faster skill acquisition is a separate experiment. For rapid exploration, run
several independent recipes/seeds on individual GPUs before introducing DDP.

Completed benchmark run IDs in `microduck-runs`:

```text
20260916-163815-bench-a100-4096-ff1289c5
20260916-163815-bench-h100-4096-bdbb93d4
20260916-163815-bench-l40s-4096-4efcd10a
20260916-164745-bench-b200-4096-4b56e75a
20260916-164810-bench-b200-16384-2599dcf1
20260916-164810-bench-h100-16384-db9f1bad
20260916-164810-bench-l40s-16384-891e76b1
```

After downloading runs, reproduce the summary with:

```bash
uv run scripts/modal/benchmark_summary.py
```

All four GPU types passed 64-env/5-update smoke training and ONNX export.
A checkpoint downloaded from H100 also passed local CPU ONNX inference.
The background-spawn jobs completed after their submitting CLI had exited.
Recovery was verified from the cancelled run
`20260916-164031-bench-b200-4096-ca656556/model_10000.pt`: a fresh B200 container
completed five further updates as
`20260916-165100-interrupted-resume-b944a851/model_10004.pt`, restoring and
advancing the curriculum counter to 240168. This exercises actual cancellation
and persistent storage, alongside the interrupted-write unit tests.

The original successful laptop policy is backed up as
`microduck-runs/local-somersault/model_9999.pt` and `local-somersault/policy.onnx`.
