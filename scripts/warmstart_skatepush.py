"""Create an explicit actor-only bootstrap for the 109D low-deck push task.

Copies the 107D policy/normalizer prefix and zero-pads the new phase columns.
Critic, optimizer, iteration and curriculum are fresh; this is not a resume.
No training occurs here. ONNX export still goes through the native exporter.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from tensordict import TensorDict
import mjlab.tasks  # noqa: F401
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


class ShapeEnv:
    num_envs = 1
    num_actions = 14
    device = "cpu"
    cfg = load_env_cfg("Mjlab-Skatepark-Push-MicroDuck")

    def get_observations(self):
        return TensorDict(
            {
                k: torch.zeros(1, n)
                for k, n in {"actor": 61, "critic": 74, "skate": 46, "push": 2}.items()
            },
            batch_size=[1],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    source = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_rl_cfg("Mjlab-Skatepark-Push-MicroDuck")
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    runner = MjlabOnPolicyRunner(ShapeEnv(), asdict(cfg), device="cpu")
    target = runner.alg.actor.state_dict()
    old = source["actor_state_dict"]
    assert (
        old["mlp.0.weight"].shape[1] == 107 and target["mlp.0.weight"].shape[1] == 109
    )
    for name, value in target.items():
        if name == "mlp.0.weight":
            value.zero_()
            value[:, :107] = old[name]
        elif name.startswith("obs_normalizer.") and value.ndim == 2:
            value[:, :107] = old[name]
        elif name == "obs_normalizer.count":
            value.fill_(10000)
        elif name == "distribution.std_param":
            value.fill_(0.35)
        else:
            value.copy_(old[name])
    runner.alg.actor.load_state_dict(target)
    output = runner.alg.save()
    output["iter"] = 0
    output["infos"] = {
        "env_state": {"common_step_counter": 0},
        "skate_curriculum": {},
        "bootstrap": {
            "source": str(a.checkpoint),
            "kind": "107D actor prefix; fresh critic/optimizer; phase weights zero",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, a.output)
    print(a.output)


if __name__ == "__main__":
    main()
