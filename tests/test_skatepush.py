from types import SimpleNamespace
import torch
import mjlab.tasks  # noqa: F401
from mjlab_microduck.robot.skateboard import (
    get_skateboard_spec,
    get_low_skateboard_spec,
    LOW_DECK_TOP,
    LOW_DECK_Z,
    LOW_WHEEL_RADIUS,
)
from mjlab_microduck.tasks.microduck_skatepush_env_cfg import (
    make_microduck_skatepush_env_cfg,
    MicroduckSkatePushRlCfg,
)
from mjlab_microduck.tasks import mdp


def test_low_board_is_passive_and_legacy_board_is_preserved():
    low = get_low_skateboard_spec().compile()
    old = get_skateboard_spec().compile()
    assert low.nu == old.nu == 0 and low.neq == 0
    assert low.njnt == old.njnt == 7
    assert 0.05 < low.body_mass.sum() < 0.09
    assert old.body_mass.sum() > 0.15
    assert LOW_DECK_TOP == 0.025 and LOW_WHEEL_RADIUS == 0.010


def test_low_config_has_explicit_phase_and_no_propulsion_assistance():
    cfg = make_microduck_skatepush_env_cfg()
    assert tuple(cfg.scene.entities) == ("robot", "board")
    assert cfg.events["reset_skateboard"].params["speed_range"] == (0.0, 0.0)
    assert cfg.events["reset_skateboard"].params["deck_z"] == LOW_DECK_Z
    assert cfg.events["reset_skateboard"].params["landing_start_prob"] == 0
    assert "push_phase" in cfg.commands and "push" in cfg.observations
    assert MicroduckSkatePushRlCfg.obs_groups["actor"] == ("actor", "skate", "push")
    assert "bad_support" in cfg.terminations
    assert cfg.rewards["push_failure"].weight < 0
    assert cfg.rewards["push_feet"].weight < 0


def test_phase_targets_are_continuous_and_are_not_motor_targets():
    phases = torch.tensor([0.0, 0.41999, 0.42001, 0.64999, 0.65, 0.65001, 0.99999])
    env = SimpleNamespace(
        num_envs=len(phases),
        device="cpu",
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(phase=phases)
        ),
        event_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(
                params={
                    "deck_z": LOW_DECK_Z,
                    "deck_top": LOW_DECK_TOP,
                    "wheel_radius": LOW_WHEEL_RADIUS,
                }
            )
        ),
        scene={
            "board_clearance": SimpleNamespace(
                data=SimpleNamespace(heights=torch.full((len(phases), 1), LOW_DECK_Z))
            )
        },
    )
    targets = mdp.skate_push_targets(env)
    assert targets.shape == (len(phases), 2, 3)
    assert torch.isfinite(targets).all()
    assert (targets[1] - targets[2]).abs().max() < 1e-4
    assert (targets[3] - targets[4]).abs().max() < 1e-4
    assert (targets[4] - targets[5]).abs().max() < 1e-4
    assert (targets[0] - targets[-1]).abs().max() < 1e-4


def test_bad_support_rejects_nonfoot_and_underside_contacts():
    zeros = torch.zeros(2, 1)
    robot = SimpleNamespace(
        find_sites=lambda names: ([0, 1], list(names)),
        data=SimpleNamespace(
            site_pos_w=torch.tensor(
                [
                    [[0.0, 0.03, 0.026], [0.0, -0.03, 0.026]],
                    [[0.0, 0.03, 0.026], [0.0, -0.03, 0.0]],
                ]
            )
        ),
    )
    board = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, LOW_DECK_Z]]).repeat(2, 1),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        )
    )
    env = SimpleNamespace(
        num_envs=2,
        scene={
            "robot": robot,
            "board": board,
            "nonfoot_board_contact": SimpleNamespace(
                data=SimpleNamespace(found=zeros.clone())
            ),
            "nonfoot_floor_contact": SimpleNamespace(
                data=SimpleNamespace(found=zeros.clone())
            ),
            "feet_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.ones(2, 2))
            ),
        },
        event_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(
                params={
                    "deck_z": LOW_DECK_Z,
                    "deck_top": LOW_DECK_TOP,
                    "wheel_radius": LOW_WHEEL_RADIUS,
                }
            )
        ),
    )
    assert mdp.skate_push_bad_contact(env).tolist() == [False, True]
    env.scene["nonfoot_board_contact"].data.found[0] = 1
    assert mdp.skate_push_bad_contact(env).tolist() == [True, True]
