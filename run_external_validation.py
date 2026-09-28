#!/usr/bin/env python3
"""Validate provenance and anti-leakage constraints before external EXP-6.

The executable intentionally contains no proprietary or fabricated
measurements.  It turns reviewer-requested external validation into a precise
gate: computation cannot begin until a cited dataset, redistribution status,
predeclared falsifiers, and component-disjoint split are present.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "experimental"
MANIFEST = DATA / "dataset_manifest.json"
MEASUREMENTS = DATA / "measurements.csv"

REQUIRED_COLUMNS = {
    "system_id",
    "component_1_inchikey",
    "component_2_inchikey",
    "temperature_K",
    "x1",
    "x2",
    "source_doi_or_record",
    "split",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not MANIFEST.exists() or not MEASUREMENTS.exists():
        raise SystemExit(
            "EXP-6 NOT RUN: provide data/experimental/dataset_manifest.json "
            "and measurements.csv from traceable experimental sources."
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("redistribution_authorized", False):
        raise SystemExit("EXP-6 NOT RUN: dataset redistribution/use authorization is not confirmed.")
    if "AUTHOR ACTION REQUIRED" in json.dumps(manifest):
        raise SystemExit("EXP-6 NOT RUN: provenance manifest still contains author-action placeholders.")
    with MEASUREMENTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("EXP-6 NOT RUN: measurements.csv contains no experimental rows.")
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise SystemExit(f"EXP-6 NOT RUN: missing required columns: {sorted(missing)}")

    component_splits: dict[str, set[str]] = {}
    for row_number, row in enumerate(rows, start=2):
        if row["split"] not in {"train", "validation", "test"}:
            raise SystemExit(f"EXP-6 NOT RUN: invalid split on CSV row {row_number}.")
        if not row["source_doi_or_record"].strip():
            raise SystemExit(f"EXP-6 NOT RUN: missing provenance on CSV row {row_number}.")
        composition = [float(row["x1"]), float(row["x2"])]
        if row.get("x3", "").strip():
            composition.append(float(row["x3"]))
        if min(composition) < 0.0 or abs(sum(composition) - 1.0) > 1.0e-6:
            raise SystemExit(f"EXP-6 NOT RUN: invalid composition on CSV row {row_number}.")
        for key in ("component_1_inchikey", "component_2_inchikey", "component_3_inchikey"):
            component = row.get(key, "").strip()
            if component:
                component_splits.setdefault(component, set()).add(row["split"])
    leakage = {component: sorted(splits) for component, splits in component_splits.items() if len(splits) > 1}
    if leakage and manifest.get("split_strategy") == "component-disjoint":
        raise SystemExit(f"EXP-6 NOT RUN: component leakage detected: {leakage}")
    if manifest.get("data_file_sha256") != sha256(MEASUREMENTS):
        raise SystemExit("EXP-6 NOT RUN: measurements.csv digest does not match the frozen manifest.")
    print(f"Schema/provenance gate passed for {len(rows)} experimental rows.")
    print("Model fitting remains disabled until the authors freeze the cited EXP-6 protocol and baselines.")


if __name__ == "__main__":
    main()

