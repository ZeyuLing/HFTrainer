#!/usr/bin/env python3
"""Runtime verification for PhysFlow MuJoCo collision/contact settings.

This is a preflight check before using MuJoCo tracking results as PhysFlow
evidence. It instantiates the same model path as run_smpl_rl_tracker.py and
asserts the contact bitmasks match the intended deployment contract:

  - body/body self-collision disabled
  - body/floor contact preserved
  - no accidental geom_gap override
  - no large MuJoCo margin override
  - passive joint forces zeroed and standard PD gear configured
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.embodied.run_smpl_rl_tracker import (  # noqa: E402
    _DEFAULT_MJCF,
    _DEFAULT_YAML,
    load_mujoco_model,
)


def _can_collide(model: mujoco.MjModel, gid_a: int, gid_b: int) -> bool:
    return bool(
        (int(model.geom_contype[gid_a]) & int(model.geom_conaffinity[gid_b]))
        or (int(model.geom_contype[gid_b]) & int(model.geom_conaffinity[gid_a]))
    )


def _geom_name(model: mujoco.MjModel, gid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return name or f"geom_{gid}"


def _load_control(yaml_path: str) -> tuple[list[float], list[float]]:
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    control = meta["control"]
    return control["stiffness"], control["damping"]


def verify(mjcf_path: str, yaml_path: str, physics_dt: float) -> int:
    stiffness, damping = _load_control(yaml_path)
    model, _ = load_mujoco_model(mjcf_path, stiffness, damping, physics_dt)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        print("FAIL floor geom named 'floor' was not found")
        return 1

    body_geoms = [
        gid for gid in range(model.ngeom)
        if gid != floor_id and int(model.geom_bodyid[gid]) > 0
    ]
    if not body_geoms:
        print("FAIL no robot body geoms found")
        return 1

    body_body_pairs = 0
    colliding_body_pairs: list[tuple[int, int]] = []
    for i, gid_a in enumerate(body_geoms):
        for gid_b in body_geoms[i + 1:]:
            body_body_pairs += 1
            if _can_collide(model, gid_a, gid_b):
                colliding_body_pairs.append((gid_a, gid_b))

    floor_contact_geoms = [
        gid for gid in body_geoms if _can_collide(model, gid, floor_id)
    ]

    failures: list[str] = []
    if colliding_body_pairs:
        sample = ", ".join(
            f"{_geom_name(model, a)}<->{_geom_name(model, b)}"
            for a, b in colliding_body_pairs[:5]
        )
        failures.append(
            f"{len(colliding_body_pairs)} body/body geom pairs can collide; sample: {sample}"
        )
    if len(floor_contact_geoms) != len(body_geoms):
        failures.append(
            f"only {len(floor_contact_geoms)}/{len(body_geoms)} body geoms can contact floor"
        )

    max_margin = float(model.geom_margin.max())
    max_gap = float(model.geom_gap.max())
    if max_margin > 0.001001:
        failures.append(f"max geom_margin is {max_margin:.6f}, expected MJCF/default <= 0.001")
    if max_gap != 0.0:
        failures.append(f"max geom_gap is {max_gap:.6f}, expected 0")

    if abs(float(model.opt.timestep) - physics_dt) > 1e-12:
        failures.append(
            f"physics timestep is {model.opt.timestep}, expected {physics_dt}"
        )

    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
        failures.append(f"integrator is {model.opt.integrator}, expected Euler")

    if float(abs(model.jnt_stiffness).max()) != 0.0:
        failures.append("passive joint stiffness is not zero")
    if float(abs(model.dof_damping).max()) != 0.0:
        failures.append("passive DOF damping is not zero")
    if float(abs(model.dof_frictionloss).max()) != 0.0:
        failures.append("DOF frictionloss is not zero")
    if float(abs(model.actuator_gear[:, 0] - 1.0).max()) > 1e-12:
        failures.append("actuator gear is not uniformly 1.0")

    print("MuJoCo collision/contact preflight")
    print(f"  mjcf: {mjcf_path}")
    print(f"  timestep: {model.opt.timestep:.6f}s")
    print(f"  integrator: {model.opt.integrator} (Euler={int(mujoco.mjtIntegrator.mjINT_EULER)})")
    print(f"  geoms: total={model.ngeom}, body={len(body_geoms)}, floor={floor_id}")
    print(f"  body/body pairs checked: {body_body_pairs}")
    print(f"  body/body colliding pairs: {len(colliding_body_pairs)}")
    print(f"  body geoms with floor contact: {len(floor_contact_geoms)}/{len(body_geoms)}")
    print(f"  geom_margin range: [{model.geom_margin.min():.6f}, {model.geom_margin.max():.6f}]")
    print(f"  geom_gap range: [{model.geom_gap.min():.6f}, {model.geom_gap.max():.6f}]")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPASS collision/contact settings match the PhysFlow MuJoCo contract")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", default=_DEFAULT_MJCF)
    parser.add_argument("--yaml", default=_DEFAULT_YAML)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    args = parser.parse_args()
    return verify(args.mjcf, args.yaml, args.physics_dt)


if __name__ == "__main__":
    raise SystemExit(main())
