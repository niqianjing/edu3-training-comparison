from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdPhysics

USD = Path("/home/zero/edu3_work/final_asset_20260716/edu3_nqj13_trainable_fullbody_v1/usd/edu3_nqj13_trainable_fullbody.usd")
stage = Usd.Stage.Open(str(USD))
if stage is None:
    raise RuntimeError(f"Cannot open {USD}")

count = 0
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue
    name = prim.GetName()
    high = "thigh_pitch" in name or "knee" in name
    coulomb = 0.51 if high else 0.146
    viscous = 0.0432 if high else 0.0306

    legacy = prim.GetAttribute("physxJoint:jointFriction")
    if not legacy:
        legacy = prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float)
    legacy.Set(0.0)

    drive_damping = prim.GetAttribute("drive:angular:physics:damping")
    if drive_damping:
        drive_damping.Set(0.0)

    prim.CreateAttribute("edu3:measuredCoulombFrictionNm", Sdf.ValueTypeNames.Float, custom=True).Set(coulomb)
    prim.CreateAttribute("edu3:measuredViscousDampingNmSecPerRad", Sdf.ValueTypeNames.Float, custom=True).Set(viscous)
    count += 1

stage.GetRootLayer().Save()
print(f"EDU3_USD_LEGACY_FRICTION_PATCH=PASS joints={count} legacy=0 drive_damping=0 explicit_si_attrs=1", flush=True)
simulation_app.close()
