"""Low-deck push curriculum: the policy learns a physical one-foot push gait.

109 actor inputs: unchanged 61+46 skate blocks plus a 2D phase cue. Phase sets
reward targets only; there are no prescribed motor actions or board forces.
"""

from copy import deepcopy
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    MetricsTermCfg,
    TerminationTermCfg,
)
from mjlab.envs.mdp import generated_commands
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab_microduck.robot.skateboard import (
    LOW_SKATEBOARD_CFG,
    LOW_DECK_Z,
    LOW_DECK_TOP,
    LOW_WHEEL_RADIUS,
)
from .microduck_skatepark_env_cfg import (
    make_microduck_skatepark_env_cfg,
    MicroduckSkateparkRlCfg,
)
from . import mdp


def make_microduck_skatepush_env_cfg(play=False, flat=False):
    cfg = make_microduck_skatepark_env_cfg(play=play, flat=flat)
    cfg.scene.entities["board"] = deepcopy(LOW_SKATEBOARD_CFG)
    # Foundation pass: hold the real head servos at HOME while learning pushes.
    # They are not welded; their BAM dynamics remain active. Later passes can
    # restore a nonzero scale once foot-only support is reliable.
    cfg.actions["joint_pos"].scale = {
        r".*(hip|knee|ankle).*": 1.0,
        r".*(head|neck).*": 0.0,
    }
    cfg.events["reset_skateboard"].params.update(
        deck_z=LOW_DECK_Z,
        deck_top=LOW_DECK_TOP,
        wheel_radius=LOW_WHEEL_RADIUS,
        speed_range=(0.0, 0.0),
        stationary_prob=1.0,
        landing_start_prob=0.0,
    )
    cfg.scene.sensors += (
        ContactSensorCfg(
            name="feet_floor_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=r"^(left_foot_collision|right_foot_collision)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
    )
    for name, secondary in (
        (
            "nonfoot_board_contact",
            ContactMatch(mode="subtree", pattern="deck", entity="board"),
        ),
        ("nonfoot_floor_contact", ContactMatch(mode="body", pattern="terrain")),
    ):
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
    cfg.commands["push_phase"] = mdp.SkatePushPhaseCfg(
        resampling_time_range=(1e8, 1e8), period_s=1.0
    )
    cfg.observations["push"] = ObservationGroupCfg(
        terms={
            "phase": ObservationTermCfg(
                func=generated_commands, params={"command_name": "push_phase"}
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="sanitize",
    )
    cfg.commands["twist"].ranges.lin_vel_x = (0.20, 0.40)
    cfg.commands["twist"].ranges.ang_vel_z = (-0.02, 0.02)
    cfg.commands["twist"].resampling_time_range = (12.0, 12.0)
    cfg.commands["twist"].rel_standing_envs = 0.05
    cfg.commands["skate_trick"].play_stage = 0
    cfg.curriculum = {}  # Flat push discovery first; promotion requires measured propulsion.
    cfg.rewards = {
        k: v
        for k, v in cfg.rewards.items()
        if k
        in ("action_rate_l2", "self_collisions", "dof_pos_limits", "head_pose_tracking")
    }
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["dof_pos_limits"].weight = -0.2
    cfg.rewards.update(
        {
            "push_feet": RewardTermCfg(func=mdp.skate_push_foot_cost, weight=-15.0),
            "push_contacts": RewardTermCfg(
                func=mdp.skate_push_contact_cost, weight=-2.0
            ),
            "push_balance": RewardTermCfg(func=mdp.skate_push_balance, weight=1.5),
            "push_drive": RewardTermCfg(func=mdp.skate_push_drive, weight=4.0),
            "push_failure": RewardTermCfg(func=mdp.skate_push_failure, weight=-10.0),
            "push_neck_rate": RewardTermCfg(
                func=mdp.skate_push_neck_cost, weight=-0.05
            ),
        }
    )
    for field in (
        "floor_contact",
        "distance",
        "phase",
        "foot_error",
        "nonfoot_failure",
        "underside_failure",
    ):
        cfg.metrics["push_" + field] = MetricsTermCfg(
            func=mdp.skate_push_metric,
            params={"field": field},
            reduce="last"
            if field in ("distance", "nonfoot_failure", "underside_failure")
            else "mean",
        )
    cfg.terminations["nan_state"].params["sensor_names"] += ("feet_floor_contact",)
    cfg.terminations["bad_support"] = TerminationTermCfg(
        func=mdp.skate_push_bad_contact
    )
    return cfg


MicroduckSkatePushRlCfg = deepcopy(MicroduckSkateparkRlCfg)
MicroduckSkatePushRlCfg.experiment_name = "microduck_skatepush"
MicroduckSkatePushRlCfg.run_name = "skatepush"
MicroduckSkatePushRlCfg.obs_groups = {
    "actor": ("actor", "skate", "push"),
    "critic": ("critic", "skate", "push"),
}
MicroduckSkatePushRlCfg.actor.distribution_cfg["init_std"] = 0.35
MicroduckSkatePushRlCfg.algorithm.entropy_coef = 0.004
