from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import mjlab.tasks  # noqa: F401, populate registry before importing robot modules

from mjlab_microduck.robot.skateboard import get_skateboard_spec
from mjlab_microduck.tasks.microduck_skatepark_env_cfg import (
    make_microduck_skatepark_env_cfg,
    MicroduckSkateparkRlCfg,
)
from mjlab_microduck.tasks.skatepark_terrain import SkateparkTerrainCfg
from mjlab_microduck.tasks import mdp


def test_board_has_no_actuators_or_foot_attachments():
    model = get_skateboard_spec().compile()
    assert model.nu == 0 and model.neq == 0
    assert model.njnt == 7  # free root, two kingpins, four wheels
    names = [model.joint(i).name for i in range(1, model.njnt)]
    assert all(n.startswith("passive_") for n in names)
    assert sum(n.endswith("wheel") for n in names) == 4
    assert all(
        model.geom_group[i] == 2
        for i in range(model.ngeom)
        if "wheel_collision" in model.geom(i).name
    )
    assert 0.1 < model.body_mass.sum() < 0.3


def test_steering_is_mechanically_mirrored():
    trajectories = []
    for torque in (-0.04, 0.04):
        spec = get_skateboard_spec()
        spec.option.timestep = 0.0025
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(10, 10, 0.1))
        spec.body("deck").add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.015, 0, 0),
            pos=(0, 0, 0.14),
            mass=0.737,
            contype=0,
            conaffinity=0,
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        data.qvel[0] = 0.5
        for j in range(model.njnt):
            if model.joint(j).name.endswith("wheel"):
                data.qvel[model.jnt_dofadr[j]] = 25
        for _ in range(400):
            data.xfrc_applied[model.body("deck").id, 3] = torque
            mujoco.mj_step(model, data)
        trajectories.append(data.qpos[:3].copy())
    assert trajectories[0][0] > 0.4
    assert trajectories[0][1] > 0.002 and trajectories[1][1] < -0.002
    np.testing.assert_allclose(trajectories[0][1], -trajectories[1][1], atol=1e-5)


@pytest.mark.parametrize(
    "kind", ["cruise", "banks", "rollers", "tabletop", "quarterpipe"]
)
def test_park_profile_has_flat_safe_spawn_and_seeded_variation(kind):
    cfg = SkateparkTerrainCfg(kind=kind, size=(10.0, 4.0))
    x, flat = cfg.profile(0.1, np.random.default_rng(42))
    assert np.all(flat == 0)
    _, a = cfg.profile(1.0, np.random.default_rng(42))
    _, b = cfg.profile(1.0, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert np.all(a[x <= 1.5] == 0)
    assert np.max(np.abs(np.diff(a) / np.diff(x))) < 0.5


def test_config_preserves_robot_contract_and_declares_extra_sensors():
    cfg = make_microduck_skatepark_env_cfg()
    assert tuple(cfg.scene.entities) == ("robot", "board")
    assert list(cfg.observations["actor"].terms)[-3:] == [
        "command",
        "head_command",
        "body_command",
    ]
    assert "skate" in cfg.observations
    assert MicroduckSkateparkRlCfg.obs_groups["actor"] == ("actor", "skate")
    assert "expand_bam_friction_fields" in cfg.events
    assert "push_robot" not in cfg.events
    assert "reset_base" not in cfg.events
    assert cfg.actions["joint_pos"].entity_name == "robot"
    assert cfg.rewards["wheel_slip"].weight < 0
    assert cfg.rewards["trick_progress"].weight > 0
    assert cfg.scene.terrain.max_init_terrain_level == 0


@pytest.mark.parametrize(
    "kind", ["cruise", "banks", "rollers", "tabletop", "quarterpipe"]
)
def test_compiled_ramp_surface_matches_profile_without_gaps(kind):
    cfg = SkateparkTerrainCfg(kind=kind, size=(10.0, 4.0))
    x, heights = cfg.profile(1.0, np.random.default_rng(42))
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg.function(1.0, spec, np.random.default_rng(42))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = np.zeros(1, dtype=np.int32)
    for sample in np.linspace(0.01, 9.99, 150):
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([sample, 2.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            1,
            -1,
            geom_id,
        )
        assert distance >= 0
        assert 1 - distance == pytest.approx(np.interp(sample, x, heights), abs=1e-5)


def test_standing_height_matches_actual_home_feet():
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
    from mjlab_microduck.robot.skateboard import ROBOT_STAND_Z

    model = MICRODUCK_STANDUP_ROBOT_CFG.build().spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for name in ("left_foot", "right_foot"):
        assert -data.site_xpos[model.site(name).id, 2] == pytest.approx(
            ROBOT_STAND_Z, abs=0.0005
        )


def test_skate_material_does_not_mutate_other_tasks():
    from mjlab.terrains import TerrainEntityCfg

    before = TerrainEntityCfg().materials[0].geom_names_expr
    make_microduck_skatepark_env_cfg()
    assert TerrainEntityCfg().materials[0].geom_names_expr == before


def fake_trick():
    def tensor(values):
        return torch.tensor([values], dtype=torch.float32)

    board = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=tensor([0, 0, 0.044]),
            root_link_quat_w=tensor([1, 0, 0, 0]),
            root_link_lin_vel_w=tensor([0.5, 0, 0]),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=tensor([0, 0, 0.166]), root_link_quat_w=tensor([1, 0, 0, 0])
        )
    )
    board.data.root_link_lin_vel_b = board.data.root_link_lin_vel_w
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        common_step_counter=0,
        _skate_reset_step=torch.zeros(1, dtype=torch.long),
        scene={
            "robot": robot,
            "board": board,
            "board_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=tensor([1, 1, 1, 1]))
            ),
            "board_support_contact": SimpleNamespace(
                data=SimpleNamespace(found=tensor([1]))
            ),
            "feet_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=tensor([1, 1]))
            ),
            "board_clearance": SimpleNamespace(
                data=SimpleNamespace(heights=tensor([0.044]))
            ),
        },
    )
    term = mdp.SkateTrickCommand(
        mdp.SkateTrickCommandCfg(resampling_time_range=(4, 4)), env
    )
    term.mode[:] = 1
    return env, term


def test_trick_requires_board_airtime_and_landing_is_one_shot():
    env, term = fake_trick()
    # Duck-only jumping cannot qualify: wheels remain grounded.
    env.scene["robot"].data.root_link_pos_w[:, 2] = 0.25
    for _ in range(5):
        env.common_step_counter += 1
        term.update_state()
    assert term.air_time.item() == 0 and term.completion.item() == 0
    # Actual board flight, then a stable supported ride-away.
    env.scene["board_ground_contact"].data.found[:] = 0
    env.scene["board_clearance"].data.heights[:] = 0.08
    # A manual with the tail scraping the floor is not airborne.
    env.common_step_counter += 1
    term.update_state()
    assert term.air_time.item() == 0
    env.scene["board_support_contact"].data.found[:] = 0
    for _ in range(4):
        env.common_step_counter += 1
        term.update_state()
    env.scene["board_ground_contact"].data.found[:] = 1
    env.scene["board_support_contact"].data.found[:] = 1
    env.scene["board_clearance"].data.heights[:] = 0.044
    env.scene["robot"].data.root_link_pos_w[:, 2] = 0.166
    payout = 0
    for _ in range(20):
        env.common_step_counter += 1
        term.update_state()
        before = term.successes.clone()
        term.update_state()  # Multiple reward consumers cannot update twice.
        assert torch.equal(before, term.successes)
        payout += term.completion.item() * env.step_dt
    assert payout == pytest.approx(1.0)
    assert term.successes.item() == 1
    assert term.command[0, :3].tolist() == [1.0, 0.0, 0.0]
    term.cfg.play_mode = "ollie"
    term._resample_command(torch.tensor([0]))
    assert term.command[0, :3].tolist() == [0.0, 1.0, 0.0]
    for _ in range(10):
        env.common_step_counter += 1
        term.update_state()
    assert term.successes.item() == 1  # A new request needs a fresh airborne phase.


def test_ground_spin_then_jump_does_not_count_as_airborne_180():
    one = torch.ones(1)
    good = dict(
        air_time=one * 0.1,
        grounded=one.bool(),
        feet_supported=one.bool(),
        aboard=one.bool(),
        robot_up=one,
        board_up=one,
        speed=one * 0.5,
        yaw_error=one * 0,
        mode=one.long() * 2,
        air_rotation=one * 0,
    )
    assert not mdp.skate_landing_valid(**good).item()
    good["air_rotation"] = one * 3.14
    assert mdp.skate_landing_valid(**good).item()
    good["feet_supported"] = ~one.bool()
    assert not mdp.skate_landing_valid(**good).item()


def test_failed_episode_cannot_promote_terrain():
    result = mdp.skate_episode_success(
        torch.tensor([12.0, 12.0, 3.0, 12.0]),
        torch.tensor([2.0, 2.0, 2.0, 0.0]),
        torch.tensor([0.5, 0.5, 0.5, 0.0]),
        torch.ones(4, dtype=torch.bool),
        torch.tensor([False, True, False, False]),
    )
    assert result.tolist() == [True, False, False, True]


def test_assisted_landings_do_not_count_as_self_initiated_tricks():
    env, term = fake_trick()
    env._skate_assisted_start = torch.ones(1, dtype=torch.bool)
    term._resample_command(torch.tensor([0]))
    assert term.assisted.item() and term.attempts.item() == 0
    term.air_time[:] = 0.1
    for _ in range(10):
        env.common_step_counter += 1
        term.update_state()
    assert term.assisted_successes.item() == 1
    assert term.successes.item() == 0
    assert (
        make_microduck_skatepark_env_cfg(play=True)
        .events["reset_skateboard"]
        .params["landing_start_prob"]
        == 0
    )


def test_quality_costs_wait_for_skill_and_have_correct_sign(monkeypatch):
    env = SimpleNamespace(_skate_progress={"trick_ema": 0.0})
    monkeypatch.setattr(
        mdp, "trunk_vertical_accel_penalty", lambda e: torch.tensor([-100.0])
    )
    monkeypatch.setattr(mdp, "joint_torque_rate_l2", lambda e: torch.tensor([200.0]))
    assert mdp.skate_quality_cost(env).item() == 0
    env._skate_progress["trick_ema"] = 0.7
    assert mdp.skate_quality_cost(env).item() == pytest.approx(100.0)
    assert mdp.skate_quality_cost(env, "torque").item() == pytest.approx(200.0)
    cfg = make_microduck_skatepark_env_cfg()
    assert cfg.rewards["skate_impact"].weight < 0
    assert cfg.rewards["skate_torque_rate"].weight < 0


def test_new_episode_goal_does_not_read_stale_previous_heading():
    env, term = fake_trick()
    env.scene["board"].data.root_link_quat_w[:] = torch.tensor(
        [[2**-0.5, 0.0, 0.0, 2**-0.5]]
    )
    term.cfg.play_mode = "180"
    term._resample_command(torch.tensor([0]))
    assert term.target_yaw.abs().item() == pytest.approx(np.pi)
    assert term.previous_yaw.item() == 0


def test_rolling_command_allows_sideways_rider_and_fakie_but_not_sliding():
    env, term = fake_trick()
    env.scene["board"].data.root_link_ang_vel_b = torch.zeros(1, 3)
    env.command_manager = SimpleNamespace(
        get_command=lambda name: torch.tensor([[0.5, 0.0, 0.0]]),
        get_term=lambda name: term,
    )
    assert mdp.skate_riding_reward(env).item() == pytest.approx(1.0)
    env.scene["board"].data.root_link_lin_vel_w[:, 0] = -0.5
    env.common_step_counter += 2
    env._skate_reset_step[:] = env.common_step_counter
    assert mdp.skate_riding_reward(env).item() == pytest.approx(1.0)
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor(
        [[2**-0.5, 0, 0, 2**-0.5]]
    )
    assert mdp.skate_riding_reward(env).item() == pytest.approx(1.0)
    env.scene["board"].data.root_link_lin_vel_w[:] = 0
    env.scene["board"].data.root_link_lin_vel_w[:, 1] = 0.5
    env.common_step_counter += 2
    env._skate_reset_step[:] = env.common_step_counter
    assert mdp.skate_riding_reward(env).item() < 0.1


def test_control_ramp_is_gradual_persistent_and_cannot_be_escaped():
    env = SimpleNamespace(common_step_counter=7000, _skate_progress={"ride_ema": 0.2})
    assert mdp.skate_control_strength(env) == 0
    assert "control_start_step" not in env._skate_progress
    env._skate_progress["ride_ema"] = 0.8
    assert mdp.skate_control_strength(env) == 0
    env.common_step_counter += 100 * 24
    assert mdp.skate_control_strength(env) == 0.5
    restored = SimpleNamespace(
        common_step_counter=env.common_step_counter,
        _skate_progress=dict(env._skate_progress),
    )
    restored._skate_progress["ride_ema"] = 0
    assert mdp.skate_control_strength(restored) == 0.5
    restored.common_step_counter += 100 * 24
    assert mdp.skate_control_strength(restored) == 1


def test_control_costs_are_name_resolved_reset_safe_and_leave_trick_freedom():
    goal = SimpleNamespace(mode=torch.tensor([0]), completed=torch.tensor([False]))
    torque = torch.zeros(1, 4)
    env = SimpleNamespace(
        common_step_counter=6000,
        _skate_reset_step=torch.tensor([0]),
        _skate_progress={"ride_ema": 0.8, "control_start_step": 1200},
        scene={"robot": SimpleNamespace(data=SimpleNamespace(actuator_force=torque))},
        action_manager=SimpleNamespace(
            action=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            prev_action=torch.zeros(1, 4),
            get_term=lambda name: SimpleNamespace(
                target_names=["hip", "head_yaw", "knee", "neck_pitch"]
            ),
        ),
        command_manager=SimpleNamespace(get_term=lambda name: goal),
    )
    assert mdp.skate_control_cost(env).item() == 30
    assert mdp.skate_control_cost(env, "neck").item() == 20
    goal.mode[:] = 1
    assert mdp.skate_control_cost(env).item() == pytest.approx(23.5)
    assert mdp.skate_control_cost(env, "neck").item() == 20
    env.common_step_counter += 1
    torque[:] = 1
    assert mdp.skate_control_terms(env)["torque"].item() == 4
    assert mdp.skate_control_terms(env)["torque"].item() == 4
    env.common_step_counter += 1
    env._skate_reset_step[:] = env.common_step_counter - 1
    torque[:] = 2
    assert mdp.skate_control_cost(env).item() == 0
    assert mdp.skate_control_cost(env, "torque").item() == 0
    env.common_step_counter += 1
    assert mdp.skate_control_terms(env)["torque"].item() == 0


def test_small_real_hops_get_shaping_without_being_credited_as_tricks():
    env, term = fake_trick()
    env.scene["board_support_contact"].data.found[:] = 0
    env.scene["board_clearance"].data.heights[:] = 0.048
    env.common_step_counter += 1
    term.update_state()
    assert term.progress.item() > 0
    assert term.air_time.item() == 0
    assert term.successes.item() == 0
    env.common_step_counter += 1
    term.update_state()
    assert term.progress.item() == 0  # Holding a small airborne gap is not an annuity.


def test_foreaft_stance_requires_supported_feet_and_saturates():
    env, term = fake_trick()
    robot = env.scene["robot"]
    robot.find_sites = lambda names: ([0, 1], list(names))
    robot.data.site_pos_w = torch.tensor([[[0.042, 0.0, 0.05], [-0.042, 0.0, 0.05]]])
    env.scene["board"].data.root_link_ang_vel_b = torch.zeros(1, 3)
    env.command_manager = SimpleNamespace(
        get_command=lambda name: torch.tensor([[0.5, 0.0, 0.0]]),
        get_term=lambda name: term,
    )
    assert mdp.skate_foreaft_stance_reward(env).item() == pytest.approx(1.0)
    env.scene["feet_ground_contact"].data.found[:, 1] = 0
    assert mdp.skate_foreaft_stance_reward(env).item() == 0
    env.scene["feet_ground_contact"].data.found[:] = 1
    robot.data.site_pos_w[:] = torch.tensor([[[0.0, 0.042, 0.05], [0.0, -0.042, 0.05]]])
    assert mdp.skate_foreaft_stance_reward(env).item() == 0


def test_ppo_random_episode_age_cannot_fake_curriculum_success():
    env = SimpleNamespace(
        common_step_counter=1001,
        step_dt=0.02,
        _skate_start_pos=torch.zeros(1, 3),
        _skate_reset_step=torch.tensor([1000]),
        reset_terminated=torch.tensor([False]),
        episode_length_buf=torch.tensor([600]),
    )
    assert mdp.skate_curriculum(env, torch.tensor([0])) == 0
    assert env._skate_progress["episodes"] == 0
    # Repeated resets and a cold exporter have no new physical episode to grade.
    env.common_step_counter = 1000
    env.reset_terminated[:] = True
    assert mdp.skate_curriculum(env, torch.tensor([0])) == 0
    assert env._skate_progress["episodes"] == 0
    del env.reset_terminated
    assert mdp.skate_curriculum(env, torch.tensor([0])) == 0


def test_pose_reward_cannot_be_farmed_by_parking_under_forward_command():
    env, term = fake_trick()
    env.scene["board"].data.root_link_ang_vel_b = torch.zeros(1, 3)
    target = torch.tensor([[0.5, 0.0, 0.0]])
    env.command_manager = SimpleNamespace(
        get_command=lambda name: target if name == "twist" else torch.zeros(1, 6),
        get_term=lambda name: term,
    )
    moving = mdp.skate_stance_reward(env).item()
    env.scene["board"].data.root_link_lin_vel_w[:] = 0
    env.common_step_counter += 2
    env._skate_reset_step[:] = env.common_step_counter
    assert mdp.skate_stance_reward(env).item() < moving * 0.1
    target[:] = 0
    assert mdp.skate_stance_reward(env).item() == pytest.approx(moving)


def test_sustained_speed_rejects_reversal_dither_and_resets_cleanly():
    env, _ = fake_trick()
    env._skate_reset_step[:] = -10
    late = []
    for step in range(120):
        env.common_step_counter += 1
        env.scene["board"].data.root_link_lin_vel_w[:, 0] = 0.5 if step % 2 else -0.5
        speed = mdp.skate_sustained_speed(env).item()
        assert mdp.skate_sustained_speed(env).item() == speed
        if step > 100:
            late.append(speed)
    assert max(late) < 0.03
    env._skate_reset_step[:] = env.common_step_counter
    env.common_step_counter += 1
    env.scene["board"].data.root_link_lin_vel_w[:, 0] = 0.4
    assert mdp.skate_sustained_speed(env).item() == pytest.approx(0.4)
    # Board turns 180 but continues travelling in the same world direction.
    env.scene["board"].data.root_link_quat_w[:] = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert mdp.skate_sustained_speed(env).item() == pytest.approx(0.4)


def test_pop_setup_shapes_progress_but_cannot_farm_or_count_as_flight():
    env, term = fake_trick()
    env.scene["board"].data.root_link_quat_w[:] = torch.tensor(
        [[np.cos(0.1), 0.0, -np.sin(0.1), 0.0]]
    )
    env.common_step_counter += 1
    term.update_state()
    assert 0 < term.progress.item() * env.step_dt <= 0.15
    assert term.air_time.item() == 0 and term.successes.item() == 0
    env.common_step_counter += 1
    term.update_state()
    assert term.progress.item() == 0
