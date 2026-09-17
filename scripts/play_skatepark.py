"""Inspect the physical park or play a learned skateboard policy.

uv run scripts/play_skatepark.py --preview
uv run scripts/play_skatepark.py --checkpoint logs/modal/<run>/model_500.pt
"""

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--preview",
        action="store_true",
        help="Constant calibrated stance; not a learned policy",
    )
    parser.add_argument(
        "--course",
        choices=("cruise", "banks", "rollers", "tabletop", "quarterpipe"),
        default="rollers",
    )
    parser.add_argument("--difficulty", type=float, default=0.35)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--task", default="Mjlab-Skatepark-MicroDuck")
    parser.add_argument(
        "--rider-yaw-deg",
        type=float,
        default=0.0,
        help="Rider orientation relative to the board at reset",
    )
    parser.add_argument(
        "--speed", type=float, help="Fix nonnegative longitudinal rolling speed"
    )
    parser.add_argument("--turn-rate", type=float, help="Fix the yaw-rate command")
    parser.add_argument(
        "--start-speed",
        type=float,
        help="Fix reset velocity; zero tests self-propulsion",
    )
    parser.add_argument(
        "--tricks", choices=("auto", "ride", "ollie", "180", "chain"), default="auto"
    )
    parser.add_argument("--viewer", choices=("native", "viser"), default="native")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Evaluate the first episode of each duck",
    )
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample training-style actions instead of the policy mean",
    )
    parser.add_argument("--report", type=Path, help="Save headless metrics as JSON")
    parser.add_argument("--video", type=Path, help="Record a headless MP4 of duck zero")
    args = parser.parse_args()
    if not 0 <= args.difficulty <= 1:
        parser.error("--difficulty must be between 0 and 1")
    if args.num_envs < 1 or args.seconds <= 0:
        parser.error("--num-envs and --seconds must be positive")
    if args.speed is not None and args.speed < 0:
        parser.error("--speed is nonnegative rolling speed (forward or fakie)")
    if args.stochastic and args.preview:
        parser.error("--stochastic requires a checkpoint")
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    task = args.task
    cfg = load_env_cfg(task, play=True)
    if "push" in cfg.observations:
        from mjlab.sensor import ContactSensorCfg, ContactMatch

        cfg.scene.sensors += (
            ContactSensorCfg(
                name="feet_wheel_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=r"^(left_foot_collision|right_foot_collision)$",
                    entity="robot",
                ),
                secondary=ContactMatch(
                    mode="geom", pattern=r".*_wheel_collision", entity="board"
                ),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
            ),
        )
        existing = {s.name for s in cfg.scene.sensors}
        for name, secondary in (
            (
                "nonfoot_board_contact",
                ContactMatch(mode="subtree", pattern="deck", entity="board"),
            ),
            ("nonfoot_floor_contact", ContactMatch(mode="body", pattern="terrain")),
        ):
            if name not in existing:
                cfg.scene.sensors += (
                    ContactSensorCfg(
                        name=name,
                        primary=ContactMatch(
                            mode="body",
                            pattern=".*",
                            entity="robot",
                            exclude=("ankle_left", "ankle_right"),
                        ),
                        secondary=secondary,
                        fields=("found", "force"),
                        reduce="netforce",
                        num_slots=1,
                    ),
                )
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    cfg.viewer.distance = 0.85
    cfg.viewer.elevation = -18
    cfg.viewer.azimuth = 90
    cfg.viewer.max_extra_envs = 0
    cfg.commands["twist"].debug_vis = False
    headless = args.headless or args.video is not None or args.report is not None
    if headless:
        cfg.auto_reset = False  # Capture terminal results before explicitly resetting.
    if args.video:
        cfg.viewer.width, cfg.viewer.height = 1280, 720
    cfg.curriculum = {}  # Keep the chosen course fixed during playback.
    generator = cfg.scene.terrain.terrain_generator
    generator.sub_terrains = {args.course: generator.sub_terrains[args.course]}
    generator.num_rows = 1
    generator.difficulty_range = (args.difficulty, args.difficulty)
    cfg.commands["skate_trick"].play_mode = args.tricks
    cfg.events["reset_skateboard"].params.update(
        rider_yaw=math.radians(args.rider_yaw_deg), sideways_prob=0.0
    )
    if args.speed is not None:
        cfg.commands["twist"].ranges.lin_vel_x = (args.speed, args.speed)
        cfg.commands["twist"].rel_standing_envs = 0.0
    if args.turn_rate is not None:
        cfg.commands["twist"].ranges.ang_vel_z = (args.turn_rate, args.turn_rate)
        cfg.commands["twist"].rel_standing_envs = 0.0
    if args.preview:
        cfg.events = {
            k: v
            for k, v in cfg.events.items()
            if k
            in (
                "reset_skateboard",
                "reset_action_history",
                "expand_bam_friction_fields",
            )
        }
        cfg.events["reset_skateboard"].params.update(
            speed_range=(0.5, 0.5), stationary_prob=0.0
        )
        actuator = cfg.scene.entities["robot"].articulation.actuators[0]
        actuator.vin_range = (7.4, 7.4)
        actuator.vin_drop_gain_range = (0.0, 0.0)
        print("PHYSICS PREVIEW: fixed stance with one rolling push at reset.")
    if args.start_speed is not None:
        cfg.events["reset_skateboard"].params.update(
            speed_range=(args.start_speed, args.start_speed), stationary_prob=0.0
        )
    agent_cfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(
            cfg, device="cuda:0", render_mode="rgb_array" if args.video else None
        ),
        clip_actions=agent_cfg.clip_actions,
    )
    try:
        if args.checkpoint:
            runner = load_runner_cls(task)(env, asdict(agent_cfg), device="cuda:0")
            runner.load(
                str(args.checkpoint), load_cfg={"actor": True}, map_location="cuda:0"
            )
            policy = runner.get_inference_policy(device="cuda:0")
            if args.stochastic:
                actor = policy

                def policy(obs):
                    return actor(obs, stochastic_output=True)
        else:

            def policy(obs):
                return torch.zeros(args.num_envs, 14, device="cuda:0")

        if headless:
            evaluate(env, policy, args)
        else:
            viewer = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
            viewer(env, policy).run()
    finally:
        env.close()


def evaluate(env, policy, args):
    import imageio.v2 as imageio
    from mjlab.utils.lab_api.math import quat_apply_inverse
    from mjlab_microduck.tasks.mdp import _skate_yaw, wrap_to_pi

    raw = env.unwrapped
    seen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    failed = seen.clone()
    tricks = torch.zeros(args.num_envs, device=env.device)
    distance = torch.zeros_like(tricks)
    world_x = torch.zeros_like(tricks)
    path_length = torch.zeros_like(tricks)
    backward_time = torch.zeros_like(tricks)
    duration = torch.zeros_like(tricks)
    yaw_integral = torch.zeros_like(tricks)
    forward_integral = torch.zeros_like(tricks)
    rolling_integral = torch.zeros_like(tricks)
    lateral_integral = torch.zeros_like(tricks)
    action_sq = torch.zeros_like(tricks)
    neck_sq = torch.zeros_like(tricks)
    accel_sq = torch.zeros_like(tricks)
    torque_sq = torch.zeros_like(tricks)
    neck_velocity_sq = torch.zeros_like(tricks)
    body_velocity_sq = torch.zeros_like(tricks)
    knee_flexion = torch.zeros_like(tricks)
    foreaft_span = torch.zeros_like(tricks)
    rider_yaw_abs = torch.zeros_like(tricks)
    floor_time = torch.zeros(args.num_envs, 2, device=env.device)
    wheel_time = torch.zeros_like(floor_time)
    floor_impulse = torch.zeros(args.num_envs, 2, 3, device=env.device)
    deck_impulse = torch.zeros_like(floor_impulse)
    nonfoot_time = torch.zeros_like(floor_time)
    support_failures = torch.zeros_like(floor_time)
    nonfoot_names = (
        list(
            dict.fromkeys(
                slot.primary_name for slot in raw.scene["nonfoot_board_contact"]._slots
            )
        )
        if "push" in raw.cfg.observations
        else []
    )
    nonfoot_body_time = torch.zeros(
        args.num_envs, len(nonfoot_names), device=env.device
    )
    samples = torch.zeros_like(tricks)
    names = raw.action_manager.get_term("joint_pos").target_names
    target_ids = raw.action_manager.get_term("joint_pos").target_ids
    action_scale = raw.action_manager.get_term("joint_pos").scale
    neck_ids = [i for i, name in enumerate(names) if "head" in name or "neck" in name]
    knee_ids = [i for i, name in enumerate(names) if "knee" in name]
    foot_ids, _ = raw.scene["robot"].find_sites(("left_foot", "right_foot"))
    previous_torque = raw.scene["robot"].data.actuator_force.clone()
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(args.video),
            fps=round(1 / raw.step_dt),
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-movflags", "+faststart"],
        )
    try:
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(round(args.seconds / raw.step_dt)):
                obs, _, done, _ = env.step(policy(obs))
                active = ~seen
                term = raw.command_manager.get_term("skate_trick")
                tricks[active] = term.successes[active]
                offset = raw.scene["board"].data.root_link_pos_w - raw._skate_start_pos
                distance[active] = offset[active, :2].norm(dim=-1)
                world_x[active] = offset[active, 0]
                duration[active] += raw.step_dt
                measured = active & (duration > 2 * raw.step_dt)
                if "push" in raw.cfg.observations:
                    floor = raw.scene["feet_floor_contact"].data
                    wheels = raw.scene["feet_wheel_contact"].data
                    floor_time[active] += (
                        floor.found.reshape(args.num_envs, 2)[active] > 0
                    ) * raw.step_dt
                    wheel_time[active] += (
                        wheels.found.reshape(args.num_envs, 2)[active] > 0
                    ) * raw.step_dt
                    floor_impulse[active] += (
                        floor.force.reshape(args.num_envs, 2, 3)[active] * raw.step_dt
                    )
                    deck_impulse[active] += (
                        raw.scene["feet_ground_contact"].data.force.reshape(
                            args.num_envs, 2, 3
                        )[active]
                        * raw.step_dt
                    )
                    for j, name in enumerate(
                        ("nonfoot_board_contact", "nonfoot_floor_contact")
                    ):
                        nonfoot_time[active, j] += (
                            raw.scene[name].data.found.reshape(args.num_envs, -1)[
                                active
                            ]
                            > 0
                        ).any(-1) * raw.step_dt
                    nonfoot_body_time[active] += (
                        raw.scene["nonfoot_board_contact"].data.found.reshape(
                            args.num_envs, -1
                        )[active]
                        > 0
                    ) * raw.step_dt
                am = raw.action_manager
                delta = (am.action - am.prev_action) * action_scale
                second_delta = (
                    am.action - 2 * am.prev_action + am.prev_prev_action
                ) * action_scale
                torque = raw.scene["robot"].data.actuator_force
                action_sq[measured] += delta[measured].square().mean(-1)
                neck_sq[measured] += delta[measured][:, neck_ids].square().mean(-1)
                accel_sq[measured] += second_delta[measured].square().mean(-1)
                torque_sq[measured] += (
                    (torque[measured] - previous_torque[measured]).square().mean(-1)
                )
                samples[measured] += 1
                robot = raw.scene["robot"]
                joint_velocity = robot.data.joint_vel[:, target_ids]
                neck_velocity_sq[measured] += (
                    joint_velocity[measured][:, neck_ids].square().mean(-1)
                )
                body_velocity_sq[measured] += (
                    robot.data.root_link_ang_vel_b[measured].square().mean(-1)
                )
                knee_flexion[measured] += (
                    robot.data.joint_pos[measured][:, target_ids][:, knee_ids]
                    .abs()
                    .mean(-1)
                )
                previous_torque.copy_(torque)
                board = raw.scene["board"]
                foot_separation = (
                    robot.data.site_pos_w[:, foot_ids[0]]
                    - robot.data.site_pos_w[:, foot_ids[1]]
                )
                span = quat_apply_inverse(board.data.root_link_quat_w, foot_separation)[
                    :, 0
                ].abs()
                foreaft_span[measured] += span[measured]
                relative_yaw = wrap_to_pi(
                    _skate_yaw(robot.data.root_link_quat_w)
                    - _skate_yaw(board.data.root_link_quat_w)
                )
                rider_yaw_abs[measured] += relative_yaw[measured].abs()
                path_length[active] += (
                    board.data.root_link_lin_vel_w[active, :2].norm(dim=-1)
                    * raw.step_dt
                )
                backward_time[active] += (
                    board.data.root_link_lin_vel_w[active, 0] < -0.02
                ) * raw.step_dt
                velocity = quat_apply_inverse(
                    raw.scene["robot"].data.root_link_quat_w,
                    board.data.root_link_lin_vel_w,
                )
                forward_integral[active] += velocity[active, 0] * raw.step_dt
                rolling_integral[active] += (
                    board.data.root_link_lin_vel_b[active, 0].abs() * raw.step_dt
                )
                lateral_integral[active] += (
                    board.data.root_link_lin_vel_b[active, 1].abs() * raw.step_dt
                )
                yaw_integral[active] += (
                    board.data.root_link_ang_vel_b[active, 2] * raw.step_dt
                )
                failed |= active & raw.reset_terminated
                if "push" in raw.cfg.observations:
                    for j, name in enumerate(
                        ("_push_nonfoot_failure", "_push_underside_failure")
                    ):
                        if hasattr(raw, name):
                            support_failures[active, j] = getattr(raw, name)[
                                active
                            ].float()
                seen |= done.bool()
                if writer:
                    frame = raw.render()
                    writer.append_data(frame[0] if frame.ndim == 4 else frame)
                ids = done.nonzero().flatten()
                if len(ids):
                    raw.reset(env_ids=ids)
                    obs = env.get_observations()
        report = {
            "checkpoint": str(args.checkpoint),
            "preview": args.preview,
            "course": args.course,
            "difficulty": args.difficulty,
            "requested_tricks": args.tricks,
            "speed_command": args.speed,
            "turn_rate_command": args.turn_rate,
            "start_speed": args.start_speed,
            "rider_yaw_deg": args.rider_yaw_deg,
            "num_envs": args.num_envs,
            "seconds": args.seconds,
            "stochastic": args.stochastic,
            "action_delta_rms_rad": (action_sq.sum() / samples.sum().clamp_min(1))
            .sqrt()
            .item(),
            "neck_action_delta_rms_rad": (neck_sq.sum() / samples.sum().clamp_min(1))
            .sqrt()
            .item(),
            "action_second_delta_rms_rad": (accel_sq.sum() / samples.sum().clamp_min(1))
            .sqrt()
            .item(),
            "torque_delta_rms_nm": (torque_sq.sum() / samples.sum().clamp_min(1))
            .sqrt()
            .item(),
            "first_episode_survival_fraction": (~failed).float().mean().item(),
            "neck_joint_speed_rms_rad_s": (
                neck_velocity_sq.sum() / samples.sum().clamp_min(1)
            )
            .sqrt()
            .item(),
            "body_angular_speed_rms_rad_s": (
                body_velocity_sq.sum() / samples.sum().clamp_min(1)
            )
            .sqrt()
            .item(),
            "mean_knee_flexion_rad": (
                knee_flexion.sum() / samples.sum().clamp_min(1)
            ).item(),
            "mean_foreaft_foot_span_m": (
                foreaft_span.sum() / samples.sum().clamp_min(1)
            ).item(),
            "mean_rider_board_yaw_abs_rad": (
                rider_yaw_abs.sum() / samples.sum().clamp_min(1)
            ).item(),
            "mean_first_episode_seconds": duration.mean().item(),
            "mean_distance_m": distance.mean().item(),
            "mean_world_x_progress_m": world_x.mean().item(),
            "mean_path_length_m": path_length.mean().item(),
            "world_x_backward_fraction": (
                backward_time / duration.clamp_min(raw.step_dt)
            )
            .mean()
            .item(),
            "mean_forward_speed": (forward_integral / duration.clamp_min(raw.step_dt))
            .mean()
            .item(),
            "mean_yaw_rate": (yaw_integral / duration.clamp_min(raw.step_dt))
            .mean()
            .item(),
            "mean_rolling_speed": (rolling_integral / duration.clamp_min(raw.step_dt))
            .mean()
            .item(),
            "mean_lateral_speed": (lateral_integral / duration.clamp_min(raw.step_dt))
            .mean()
            .item(),
            "mean_completed_tricks": tricks.mean().item(),
            "two_trick_chain_fraction": (tricks >= 2).float().mean().item(),
        }
        print(json.dumps(report, indent=2))
        if "push" in raw.cfg.observations:
            report.update(
                floor_contact_fraction=(
                    floor_time / duration[:, None].clamp_min(raw.step_dt)
                )
                .mean(0)
                .tolist(),
                wheel_contact_fraction=(
                    wheel_time / duration[:, None].clamp_min(raw.step_dt)
                )
                .mean(0)
                .tolist(),
                floor_impulse_raw_ns=floor_impulse.mean(0).tolist(),
                deck_impulse_raw_ns=deck_impulse.mean(0).tolist(),
            )
            report["nonfoot_contact_fraction"] = (
                (nonfoot_time / duration[:, None].clamp_min(raw.step_dt))
                .mean(0)
                .tolist()
            )
            report["support_failure_fraction"] = support_failures.mean(0).tolist()
            fractions = (
                (nonfoot_body_time / duration[:, None].clamp_min(raw.step_dt))
                .mean(0)
                .tolist()
            )
            report["nonfoot_board_by_body"] = {
                name: value
                for name, value in zip(nonfoot_names, fractions)
                if value > 0
            }
            print(
                json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "floor_contact_fraction",
                            "wheel_contact_fraction",
                            "floor_impulse_raw_ns",
                            "deck_impulse_raw_ns",
                        )
                    },
                    indent=2,
                )
            )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
