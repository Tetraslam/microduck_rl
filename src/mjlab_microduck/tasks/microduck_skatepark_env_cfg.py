"""Ride → varied terrain → ollie → airborne 180 → chained requests.

Simulation-only skateboard policy: canonical 61D actor block + a 46D skate
group (26 board/rider, 15 terrain rays, 5 goal values). Native play handles
both groups; the 107D ONNX is intentionally rejected by the hardware publisher.
Only the duck is actuated. Every request must end aboard a rolling board.
"""

from copy import deepcopy
from mjlab.envs.mdp import generated_commands

from mjlab.managers import (
    CurriculumTermCfg,
    MetricsTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_STANDUP_ROBOT_CFG,
    HOME_FRAME,
)
from mjlab_microduck.robot.skateboard import SKATEBOARD_CFG
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from mjlab_microduck.tasks.skatepark_terrain import (
    SkateparkTerrainCfg,
    SkateTerrainPattern,
)

ENABLE_TRICKS = True
PARK_SIZE = (10.0, 4.0)
PARK_SEED = 42
CONTROL_RAMP_ITERATIONS = 200


def make_microduck_skatepark_env_cfg(play=False, flat=False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    cfg.scene.entities = {
        "robot": deepcopy(MICRODUCK_STANDUP_ROBOT_CFG),
        "board": deepcopy(SKATEBOARD_CFG),
    }
    # Static BAM load compensation, measured with 3 s holds on the free board.
    # Initial physical pose stays HOME; only the servo target offsets change.
    cfg.actions["joint_pos"].use_default_offset = False
    cfg.actions["joint_pos"].offset = deepcopy(HOME_FRAME.joint_pos)
    cfg.actions["joint_pos"].offset[r".*left_hip_pitch.*"] += 0.072
    cfg.actions["joint_pos"].offset[r".*right_hip_pitch.*"] -= 0.072
    cfg.actions["joint_pos"].offset[r".*left_ankle.*"] += 0.044
    cfg.actions["joint_pos"].offset[r".*right_ankle.*"] -= 0.044
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=r"^(left_foot_collision|right_foot_collision)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="deck", entity="board"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="board_ground_contact",
            primary=ContactMatch(
                mode="geom", pattern=r".*_wheel_collision", entity="board"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="netforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="board_support_contact",
            primary=ContactMatch(mode="subtree", pattern="deck", entity="board"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="netforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
            secondary=ContactMatch(
                mode="subtree", pattern="trunk_base", entity="robot"
            ),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
        TerrainHeightSensorCfg(
            name="skate_scan",
            frame=ObjRef(type="body", name="deck", entity="board"),
            pattern=SkateTerrainPattern(size=(0.8, 0.4), resolution=0.2),
            ray_alignment="yaw",
            max_distance=2.0,
            include_geom_groups=(0,),
            reduction="none",
        ),
        TerrainHeightSensorCfg(
            name="board_clearance",
            frame=ObjRef(type="body", name="deck", entity="board"),
            pattern=SkateTerrainPattern(size=(0.0, 0.0), resolution=0.1),
            ray_alignment="yaw",
            max_distance=2.0,
            include_geom_groups=(0,),
        ),
    )
    cfg.episode_length_s = 12.0
    cfg.sim.nconmax = 100
    cfg.sim.njmax = 500
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 30
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -22
    cfg.viewer.azimuth = 120
    if not flat:
        cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            max_init_terrain_level=None if play else 0,
            terrain_generator=TerrainGeneratorCfg(
                seed=PARK_SEED,
                size=PARK_SIZE,
                num_rows=5,
                num_cols=5,
                curriculum=True,
                sub_terrains={
                    kind: SkateparkTerrainCfg(kind=kind)
                    for kind in (
                        "cruise",
                        "banks",
                        "rollers",
                        "tabletop",
                        "quarterpipe",
                    )
                },
            ),
        )
        cfg.scene.terrain.materials = deepcopy(cfg.scene.terrain.materials)
        # TerrainGenerator renames every generated geom to terrain_<index>.
        cfg.scene.terrain.materials[0].geom_names_expr = ("terrain_.*",)
    # The joint/root reset is a single coherent operation across both entities.
    for name in (
        "reset_base",
        "reset_robot_joints",
        "randomize_base_orientation",
        "push_robot",
    ):
        cfg.events.pop(name, None)
    cfg.events = {
        "reset_skateboard": EventTermCfg(
            func=mdp.reset_skateboard,
            mode="reset",
            params={"landing_start_prob": 0.0 if play else 0.2, "sideways_prob": 0.0},
        ),
        **cfg.events,
    }
    cfg.curriculum = {"skate_stage": CurriculumTermCfg(func=mdp.skate_curriculum)}
    for field in ("aboard", "speed", "raw_speed", "ride_ema", "trick_ema", "chain"):
        cfg.metrics[f"skate_{field}"] = MetricsTermCfg(
            func=mdp.skate_metric,
            params={"field": field},
            reduce="last" if field == "chain" else "mean",
        )
    if not flat:
        cfg.metrics["skate_terrain_level"] = MetricsTermCfg(
            func=mdp.skate_metric, params={"field": "terrain_level"}, reduce="last"
        )
    for group in ("actor", "critic"):
        cfg.observations[group].nan_policy = "sanitize"
        cfg.observations[group].terms.pop("foot_height", None)
    cfg.observations["skate"] = ObservationGroupCfg(
        terms={
            "board": ObservationTermCfg(func=mdp.skate_board_observation),
            "terrain": ObservationTermCfg(func=mdp.skate_terrain_observation),
            "goal": ObservationTermCfg(
                func=generated_commands, params={"command_name": "skate_trick"}
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="sanitize",
    )
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (0.25, 0.65)
    twist.ranges.lin_vel_y = (-0.001, 0.001)
    twist.ranges.ang_vel_z = (-0.08, 0.08)
    twist.rel_standing_envs = 0.1
    twist.rel_turn_in_place_envs = 0.0
    twist.resampling_time_range = (4.0, 7.0)
    cfg.commands["body_pose"].ranges = (
        (-0.001, 0.001),
        (-0.001, 0.001),
        (-0.02, 0.01),
        (-0.01, 0.01),
        (-0.01, 0.01),
        (-0.01, 0.01),
    )
    cfg.commands["skate_trick"] = mdp.SkateTrickCommandCfg(
        resampling_time_range=(3.5, 5.0),
        play_stage=None if ENABLE_TRICKS else 0,
    )
    keep = {"action_rate_l2", "dof_pos_limits", "self_collisions", "head_pose_tracking"}
    cfg.rewards = {k: v for k, v in cfg.rewards.items() if k in keep}
    cfg.rewards["action_rate_l2"].weight = -0.03
    cfg.rewards["head_pose_tracking"].weight = 0.15
    cfg.rewards["self_collisions"].weight = -0.1
    cfg.rewards.update(
        {
            "ride": RewardTermCfg(func=mdp.skate_riding_reward, weight=4.0),
            "aboard": RewardTermCfg(func=mdp.skate_stance_reward, weight=2.0),
            "wheel_slip": RewardTermCfg(func=mdp.skate_slip_cost, weight=-0.1),
            "trick_progress": RewardTermCfg(
                func=mdp.skate_trick_reward, weight=0.25, params={"kind": "progress"}
            ),
            "land_and_roll": RewardTermCfg(func=mdp.skate_trick_reward, weight=4.0),
            "skate_impact": RewardTermCfg(func=mdp.skate_quality_cost, weight=-0.003),
            "skate_torque_rate": RewardTermCfg(
                func=mdp.skate_quality_cost, weight=-0.0002, params={"kind": "torque"}
            ),
            "control_action_rate": RewardTermCfg(
                func=mdp.skate_control_cost,
                weight=-0.07,
                params={"kind": "action", "ramp_iterations": CONTROL_RAMP_ITERATIONS},
            ),
            "control_neck_rate": RewardTermCfg(
                func=mdp.skate_control_cost,
                weight=-0.10,
                params={"kind": "neck", "ramp_iterations": CONTROL_RAMP_ITERATIONS},
            ),
            "control_torque_rate": RewardTermCfg(
                func=mdp.skate_control_cost,
                weight=-0.02,
                params={"kind": "torque", "ramp_iterations": CONTROL_RAMP_ITERATIONS},
            ),
            "foreaft_stance": RewardTermCfg(
                func=mdp.skate_foreaft_stance_reward, weight=0.0
            ),
        }
    )
    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["off_board"] = TerminationTermCfg(func=mdp.skate_fallen)
    cfg.terminations["course_exit"] = TerminationTermCfg(
        func=mdp.skate_course_exit, time_out=True
    )
    cfg.terminations["nan_state"].params["sensor_names"] = (
        "feet_ground_contact",
        "board_ground_contact",
        "board_support_contact",
        "self_collision",
    )
    return cfg


MicroduckSkateparkRlCfg = deepcopy(MicroduckRlCfg)
MicroduckSkateparkRlCfg.experiment_name = "microduck_skatepark"
MicroduckSkateparkRlCfg.run_name = "skatepark"
MicroduckSkateparkRlCfg.obs_groups = {
    "actor": ("actor", "skate"),
    "critic": ("critic", "skate"),
}
MicroduckSkateparkRlCfg.actor.distribution_cfg["init_std"] = 0.3
MicroduckSkateparkRlCfg.algorithm.symmetry_cfg = None
MicroduckSkateparkRlCfg.algorithm.entropy_coef = 0.002
MicroduckSkateparkRlCfg.max_iterations = 10000
MicroduckSkateparkRlCfg.save_interval = 50
