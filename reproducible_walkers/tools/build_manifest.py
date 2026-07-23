#!/usr/bin/env python3
"""Create or verify the SHA-256 inventory for reproducible_walkers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.json"
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and not any(part in IGNORED_PARTS for part in rel.parts)
        and path.suffix.lower() not in IGNORED_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory() -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted((p for p in ROOT.rglob("*") if included(p)), key=lambda p: p.as_posix()):
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    current = inventory()
    if args.verify:
        expected = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if expected != current:
            print("FAIL: SHA256SUMS.json does not match the archive")
            return 1
        print(f"PASS: {current['file_count']} files, {current['total_bytes']} bytes")
        return 0
    OUTPUT.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE: {OUTPUT} ({current['file_count']} files, {current['total_bytes']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

