import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

ROOT = Path("/home/zero/edu3_work/final_asset_20260716/edu3_nqj13_trainable_fullbody_v1")
manifest = json.loads((ROOT / "asset_manifest.json").read_text(encoding="utf-8"))
stage = Usd.Stage.Open(str(ROOT / "usd" / "edu3_nqj13_trainable_fullbody.usd"))
if stage is None:
    raise RuntimeError("Cannot open final USD")

prims = list(stage.Traverse())
rigid = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
joints = {p.GetName(): p for p in prims if p.IsA(UsdPhysics.RevoluteJoint)}
masses = []
for prim in rigid:
    value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
    if value is not None:
        masses.append(float(value))

errors = []
if len(rigid) != manifest["links"]:
    errors.append(f"rigid_bodies {len(rigid)} != {manifest['links']}")
if len(joints) != manifest["joints"]:
    errors.append(f"joints {len(joints)} != {manifest['joints']}")
if not math.isclose(sum(masses), manifest["mass_kg"], rel_tol=0.0, abs_tol=1.0e-5):
    errors.append(f"mass {sum(masses)} != {manifest['mass_kg']}")

for name, expected in manifest["limits"].items():
    prim = joints.get(name)
    if prim is None:
        errors.append(f"missing joint {name}")
        continue
    actual = {
        "lower": math.radians(float(prim.GetAttribute("physics:lowerLimit").Get())),
        "upper": math.radians(float(prim.GetAttribute("physics:upperLimit").Get())),
        "effort": float(prim.GetAttribute("drive:angular:physics:maxForce").Get()),
        "velocity": math.radians(float(prim.GetAttribute("physxJoint:maxJointVelocity").Get())),
        "legacy": float(prim.GetAttribute("physxJoint:jointFriction").Get()),
        "drive_damping": float(prim.GetAttribute("drive:angular:physics:damping").Get()),
        "coulomb": float(prim.GetAttribute("edu3:measuredCoulombFrictionNm").Get()),
        "viscous": float(prim.GetAttribute("edu3:measuredViscousDampingNmSecPerRad").Get()),
    }
    for key in ("lower", "upper"):
        if not math.isclose(actual[key], expected[key], rel_tol=0.0, abs_tol=1.0e-5):
            errors.append(f"{name} {key} {actual[key]} != {expected[key]}")
    for key in ("effort", "velocity"):
        if not math.isclose(actual[key], expected[key], rel_tol=0.0, abs_tol=1.0e-4):
            errors.append(f"{name} {key} {actual[key]} != {expected[key]}")
    if abs(actual["legacy"]) > 1.0e-8 or abs(actual["drive_damping"]) > 1.0e-8:
        errors.append(f"{name} legacy={actual['legacy']} drive_damping={actual['drive_damping']}")
    if not math.isclose(actual["coulomb"], expected["coulomb_friction_nm"], rel_tol=0.0, abs_tol=1.0e-6):
        errors.append(f"{name} coulomb {actual['coulomb']} != {expected['coulomb_friction_nm']}")
    if not math.isclose(actual["viscous"], expected["viscous_damping_nm_s_per_rad"], rel_tol=0.0, abs_tol=1.0e-6):
        errors.append(f"{name} viscous {actual['viscous']} != {expected['viscous_damping_nm_s_per_rad']}")

if errors:
    print(json.dumps({"status": "FAIL", "errors": errors}, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(2)

print(
    json.dumps(
        {
            "status": "PASS",
            "links": len(rigid),
            "joints": len(joints),
            "mass_kg": sum(masses),
            "limits_effort_velocity": "PASS",
            "legacy_physx_friction": 0.0,
            "usd_drive_damping": 0.0,
            "measured_si_attributes": "PASS",
        },
        ensure_ascii=False,
    ),
    flush=True,
)
simulation_app.close()
