"""Seeded skate lanes with continuous ramp surfaces and clear runouts.

No gaps in the first park: cruise, pump rollers, banks and tabletop transitions.
The lowest row is flat; higher rows introduce features, not taller spawn drops.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)
from mjlab.sensor import GridPatternCfg


@dataclass
class SkateTerrainPattern(GridPatternCfg):
    def generate_rays(self, mj_model, device):
        offsets, directions = super().generate_rays(mj_model, device)
        # Cast from above approaching ramps, not from inside their solid volume.
        offsets[:, 2] = 1.0
        return offsets, directions


@dataclass(kw_only=True)
class SkateparkTerrainCfg(SubTerrainCfg):
    kind: str = "cruise"

    def profile(self, difficulty, rng):
        x = np.linspace(0, self.size[0], 49)
        d = max(0.0, (difficulty - 0.2) / 0.8)
        amplitude = d * rng.uniform(0.045, 0.10)
        # A small downhill entry supplies gravitational energy to passive wheels.
        descent = -amplitude * np.clip((x - 1.5) / 2, 0, 1)
        if self.kind == "rollers":
            z = (
                descent
                + amplitude * np.sin(np.clip((x - 2) / 4, 0, 1) * 4 * np.pi) ** 2
            )
        elif self.kind == "tabletop":
            z = descent + amplitude * (
                np.clip((x - 3) / 0.8, 0, 1) - np.clip((x - 5) / 0.8, 0, 1)
            )
        elif self.kind == "banks":
            z = descent + amplitude * (
                np.clip((x - 3) / 1.5, 0, 1) - np.clip((x - 5) / 1.5, 0, 1)
            )
        elif self.kind == "quarterpipe":
            z = descent + 2.5 * amplitude * (
                1 - np.cos(np.clip((x - 5) / 2, 0, 1) * np.pi / 2)
            )
        elif self.kind == "cruise":
            z = descent
        else:
            raise ValueError(self.kind)
        return x, z

    def function(self, difficulty, spec, rng):
        body = spec.body("terrain")
        x, z = self.profile(difficulty, rng)
        geometries = []
        palette = {
            "cruise": (0.33, 0.50, 0.55, 1),
            "rollers": (0.38, 0.40, 0.59, 1),
            "banks": (0.48, 0.38, 0.56, 1),
            "tabletop": (0.46, 0.53, 0.43, 1),
            "quarterpipe": (0.52, 0.40, 0.33, 1),
        }
        for i in range(len(x) - 1):
            dx, dz = x[i + 1] - x[i], z[i + 1] - z[i]
            slope = math.atan2(dz, dx)
            thickness = 0.10
            geom = body.add_geom(
                name=f"skate_surface_{len(body.geoms)}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(
                    (x[i] + x[i + 1]) / 2 + thickness / 2 * math.sin(slope),
                    self.size[1] / 2,
                    (z[i] + z[i + 1]) / 2 - thickness / 2 * math.cos(slope),
                ),
                size=(math.hypot(dx, dz) / 2, self.size[1] / 2, thickness / 2),
                quat=(math.cos(slope / 2), 0, -math.sin(slope / 2), 0),
                rgba=palette[self.kind],
                friction=(1.0, 0.005, 0.0001),
                group=0,
            )
            geometries.append(TerrainGeometry(geom=geom, color=palette[self.kind]))
        # Lane-edge paint gives scale and travel cues without collision obstacles.
        for y in (0.08, self.size[1] - 0.08):
            for i in range(0, len(x) - 1, 4):
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=(x[i] + 0.04, y, z[i] + 0.003),
                    size=(0.035, 0.018, 0.001),
                    rgba=(1, 0.78, 0.2, 1),
                    contype=0,
                    conaffinity=0,
                    group=2,
                )
                geometries.append(TerrainGeometry(geom=geom, color=(1, 0.78, 0.2, 1)))
        return TerrainOutput(
            origin=np.array([0.8, self.size[1] / 2, 0.0]), geometries=geometries
        )
