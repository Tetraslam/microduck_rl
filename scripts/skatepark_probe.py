"""Measure a noisy HOME hold and coherent rolling entry under the real BAM model.

uv run scripts/skatepark_probe.py
Reports tilt as well as height; disabled terminations prevent resets hiding falls.
"""

import json
from pathlib import Path

import torch
import mjlab.tasks  # noqa: F401, populate registry before importing robot modules
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_skatepark_env_cfg import (
    make_microduck_skatepark_env_cfg,
)
from mjlab_microduck.tasks import mdp


def probe(speed, landing=False):
    cfg = make_microduck_skatepark_env_cfg(flat=True)
    cfg.scene.num_envs = 16
    cfg.seed = 42
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.events = {
        k: v
        for k, v in cfg.events.items()
        if k
        in ("reset_skateboard", "reset_action_history", "expand_bam_friction_fields")
    }
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator.vin_range = (7.4, 7.4)
    actuator.vin_drop_gain_range = (0.0, 0.0)
    cfg.events["reset_skateboard"].params.update(
        speed_range=(speed, speed),
        stationary_prob=0.0,
        landing_start_prob=1.0 if landing else 0.0,
    )
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    if landing:
        env._skate_progress = dict(
            stage=2, ride_ema=1.0, trick_ema=0.0, episodes=0, landed=0.0
        )
    env.reset()
    start = env.scene["board"].data.root_link_pos_w.clone()
    for _ in range(150):
        env.step(torch.zeros(16, 14, device="cuda:0"))
    robot, board = env.scene["robot"], env.scene["board"]
    quat = robot.data.root_link_quat_w
    tilt = (
        torch.acos((1 - 2 * quat[:, 1:3].square().sum(-1)).clamp(-1, 1))
        * 180
        / torch.pi
    )
    result = {
        "speed": speed,
        "assisted_landing_start": landing,
        "tilt_degrees_median": tilt.median().item(),
        "tilt_degrees_max": tilt.max().item(),
        "aboard_fraction": mdp.skate_is_aboard(env).float().mean().item(),
        "both_feet_on_deck_fraction": mdp._skate_contacts(env, "feet_ground_contact")
        .all(-1)
        .float()
        .mean()
        .item(),
        "board_z_median": board.data.root_link_pos_w[:, 2].median().item(),
        "board_travel_x_median": (board.data.root_link_pos_w[:, 0] - start[:, 0])
        .median()
        .item(),
        "board_speed_x_median": board.data.root_link_lin_vel_b[:, 0].median().item(),
        "robot_relative_position": mdp._skate_relative(env)
        .median(dim=0)
        .values.tolist(),
        "finite": bool(torch.isfinite(env.sim.data.qpos).all()),
        "assisted_landings_mean": env.command_manager.get_term("skate_trick")
        .assisted_successes.mean()
        .item(),
        "unassisted_tricks_mean": env.command_manager.get_term("skate_trick")
        .successes.mean()
        .item(),
    }
    before = env.sim.data.qpos.clone()
    mdp.reset_skateboard(
        env,
        torch.tensor([0, 3], device=env.device),
        speed_range=(speed, speed),
        noise=0.0,
        stationary_prob=0.0,
    )
    mask = torch.ones(16, dtype=torch.bool, device=env.device)
    mask[[0, 3]] = False
    result["reset_isolated"] = bool(torch.equal(before[mask], env.sim.data.qpos[mask]))
    once = env.sim.data.qpos.clone()
    mdp.reset_skateboard(
        env,
        torch.tensor([0, 3], device=env.device),
        speed_range=(speed, speed),
        noise=0.0,
        stationary_prob=0.0,
    )
    result["reset_non_accumulating"] = bool(torch.equal(once, env.sim.data.qpos))
    env.close()
    return result


if __name__ == "__main__":
    results = [probe(0.0), probe(0.5), probe(0.5, landing=True)]
    path = Path("logs/skatepark-physics.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
