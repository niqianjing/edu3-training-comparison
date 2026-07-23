"""Compare EDU3 Isaac and MuJoCo pulse outputs at 5 ms and 20 ms."""

import argparse
import json
from pathlib import Path


THRESHOLDS = {
    "5": {"q_delta_rad": 0.002, "velocity_rad_s": 0.30},
    "20": {"q_delta_rad": 0.010, "velocity_rad_s": 0.50},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", required=True)
    parser.add_argument("--mujoco", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    isaac = json.loads(Path(args.isaac).read_text(encoding="utf-8"))
    mujoco = json.loads(Path(args.mujoco).read_text(encoding="utf-8"))
    if set(isaac["joint_names"]) != set(mujoco["joint_names"]):
        raise RuntimeError("Joint sets differ")
    if isaac.get("contract_sha256") != mujoco.get("contract_sha256"):
        raise RuntimeError("Contract SHA256 differs between engines")

    rows = {}
    passed = True
    maxima = {time: {field: 0.0 for field in threshold} for time, threshold in THRESHOLDS.items()}
    for name in isaac["joint_names"]:
        rows[name] = {}
        for time, threshold in THRESHOLDS.items():
            a = isaac["results"][name][time]
            b = mujoco["results"][name][time]
            differences = {field: abs(float(a[field]) - float(b[field])) for field in threshold}
            time_pass = all(differences[field] <= threshold[field] for field in threshold)
            passed = passed and time_pass
            for field, value in differences.items():
                maxima[time][field] = max(maxima[time][field], value)
            rows[name][time] = {"isaac": a, "mujoco": b, "abs_difference": differences, "pass": time_pass}

    report = {
        "status": "PASS" if passed else "FAIL",
        "contract_sha256": isaac["contract_sha256"],
        "thresholds": THRESHOLDS,
        "max_abs_difference": maxima,
        "joint_count": len(rows),
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "joint_count", "thresholds", "max_abs_difference")}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()


