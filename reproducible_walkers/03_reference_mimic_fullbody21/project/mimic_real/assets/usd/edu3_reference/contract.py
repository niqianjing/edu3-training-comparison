"""Load and verify the EDU3 reference-walking single-source contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONTRACT_PATH = Path("/home/zero/edu3_reference_mimic_v1/contract/edu3_reference_contract_v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("schema") != "edu3.reference-walking.contract.v1":
        raise RuntimeError(f"Unsupported EDU3 contract schema: {contract.get('schema')}")
    if contract["robot"]["joint_count"] != len(contract["robot"]["joint_order"]):
        raise RuntimeError("EDU3 contract joint_count differs from joint_order length")
    if set(contract["robot"]["joint_order"]) != set(contract["robot"]["joints"]):
        raise RuntimeError("EDU3 contract joint_order differs from joint parameter keys")

    for key in ("official_manifest", "urdf", "source_mjcf", "usd", "motion", "retarget_report"):
        item = contract["provenance"][key]
        path = Path(item["path"])
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"EDU3 contract hash mismatch: {key}: expected={item['sha256']} actual={actual}")

    print(
        "EDU3_SINGLE_SOURCE_CONTRACT=PASS "
        f"version={contract['version']} "
        f"sha256={_sha256(CONTRACT_PATH)} "
        f"joints={contract['robot']['joint_count']} "
        f"actor_input={contract['control']['actor_input_dim']} "
        f"actor_output={contract['control']['actor_output_dim']}",
        flush=True,
    )
    return contract


CONTRACT = load_contract()


__all__ = ["CONTRACT", "CONTRACT_PATH", "load_contract"]
