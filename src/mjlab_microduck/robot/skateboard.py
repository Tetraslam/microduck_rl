"""Unpowered 30 cm cruiser: four bearings and two sprung, inclined kingpins."""

import math

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

WHEEL_RADIUS = 0.020
DECK_Z = 0.044
DECK_TOP = DECK_Z + 0.005
ROBOT_STAND_Z = 0.11713  # HOME foot-site height measured on groundcontact model.


def get_skateboard_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    deck = spec.worldbody.add_body(name="deck", pos=(0, 0, DECK_Z))
    deck.add_freejoint(name="floating_base_joint")
    deck.add_geom(
        name="deck_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.112, 0.074, 0.005),
        mass=0.085,
        rgba=(0.52, 0.25, 0.8, 1),
        friction=(1.5, 0.005, 0.0001),
        condim=3,
        priority=2,
        group=2,  # Physical primitives also provide the visible board geometry.
    )
    deck.add_geom(
        name="grip",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.110, 0.072, 0.0003),
        pos=(0, 0, 0.0051),
        mass=0,
        rgba=(0.08, 0.10, 0.14, 1),
        contype=0,
        conaffinity=0,
        group=2,
    )
    for sign, end in ((1, "front"), (-1, "rear")):
        angle = -sign * 0.20
        deck.add_geom(
            name=f"{end}_kicktail",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.023, 0.071, 0.005),
            pos=(sign * 0.133, 0, 0.0045),
            quat=(math.cos(angle / 2), 0, math.sin(angle / 2), 0),
            mass=0.009,
            rgba=(0.60, 0.32, 0.86, 1),
            group=2,
            friction=(1.5, 0.005, 0.0001),
            condim=3,
            priority=2,
        )
        truck = deck.add_body(name=f"{end}_truck", pos=(sign * 0.092, 0, -0.017))
        truck.add_joint(
            name=f"passive_{end}_kingpin",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=(0.70710678, 0, sign * 0.70710678),
            limited=True,
            range=(-0.22, 0.22),
            stiffness=0.7,
            damping=0.025,
            armature=0.00002,
        )
        truck.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.010, 0.060, 0.004),
            mass=0.018,
            rgba=(0.65, 0.68, 0.73, 1),
            contype=0,
            conaffinity=0,
            group=2,
        )
        for side, y in (("left", 0.076), ("right", -0.076)):
            name = f"{end}_{side}_wheel"
            wheel = truck.add_body(name=name, pos=(0, y, -0.007))
            wheel.add_joint(
                name=f"passive_{name}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=(0, 1, 0),
                damping=0.00001,
                frictionloss=0.00001,
            )
            wheel.add_geom(
                name=f"{name}_collision",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(WHEEL_RADIUS, 0.009, 0),
                quat=(0.70710678, 0.70710678, 0, 0),
                mass=0.009,
                rgba=(1.0, 0.55, 0.10, 1),
                friction=(1.0, 0.002, 0.00001),
                condim=3,
                priority=1,
                group=2,
            )
            spec.add_exclude(bodyname1="deck", bodyname2=name)
    return spec


SKATEBOARD_CFG = EntityCfg(
    spec_fn=get_skateboard_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0, 0, DECK_Z)),
    articulation=EntityArticulationInfoCfg(actuators=()),
)
