#!/usr/bin/env python3
"""Reproduce the trace-feed failure diagnosis and recover only failed records."""
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase_recovery import recover_two_phase, tpd_diagnostics
from src.thermo import PairPotentialSpec, nrtl_g_grad_lngamma, pair_potential_predict
from experiments.run_all import ternary_parameters


def energy_function(predict):
    def evaluate(x):
        x = np.asarray(x, float)
        g, _, lng = predict(x[None, :])
        return float(np.dot(x, np.log(x)) + g[0]), np.log(x) + lng[0]
    return evaluate


def main():
    source = ROOT / "results/ternary_phase_classification.json"
    model_path = ROOT / "results/ternary_revision_models.npz"
    original = json.loads(source.read_text())
    archive = np.load(model_path)
    spec = PairPotentialSpec(archive["embeddings"], hidden_units=20,
                             temperature_center=323.15, temperature_scale=50.)
    functions = {
        "reference": energy_function(lambda x: nrtl_g_grad_lngamma(x, 323.15, *ternary_parameters())),
        "model": energy_function(lambda x: pair_potential_predict(archive["potential_parameters"][0], spec, x, 323.15)),
    }
    report = {"scope": "Corrective numerical rerun, fixed feed and frozen models; no retraining or global certificate.",
              "temperature_K": 323.15, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "recovered_records": []}
    old_floor = 2e-6
    for i, record in enumerate(original["records"]):
        if record["reference"]["success"] and record["model"]["success"]:
            continue
        feed = np.asarray(record["feed"])
        row = {"zero_based_index": i, "feed": feed.tolist(), "legacy_phase_floor": old_floor,
               "components_below_legacy_floor": np.flatnonzero(feed < old_floor).tolist(),
               "infeasibility_reason": "If both phase compositions are >= eps, their convex combination is >= eps; feed violates this necessary condition.",
               "solutions": {}}
        for name, function in functions.items():
            begin = perf_counter()
            recovered = recover_two_phase(feed, function, random_starts=32, seed=947)
            if not recovered["success"]:
                raise RuntimeError(f"No accepted recovery: record {i}, {name}")
            recovered["tpd_checks"] = [
                {"phase": label, "random_starts": budget,
                 **tpd_diagnostics(np.asarray(recovered[label]), function, random_starts=budget, seed=1947)}
                for label in ("phase_a", "phase_b") for budget in (50, 100, 200)]
            recovered["elapsed_seconds_not_a_speed_benchmark"] = perf_counter()-begin
            if any(d["minimum_all_observed"] < -1e-8 for d in recovered["tpd_checks"]):
                raise RuntimeError(f"Negative stability diagnostic: record {i}, {name}")
            row["solutions"][name] = recovered
            print(name, json.dumps({k: v for k, v in recovered.items() if k not in ("start_records", "tpd_checks")}), flush=True)
        report["recovered_records"].append(row)
    corrected = json.loads(json.dumps(original))
    for row in report["recovered_records"]:
        target = corrected["records"][row["zero_based_index"]]
        target["legacy_results"] = {name: target[name] for name in ("reference", "model")}
        for name in ("reference", "model"):
            target[name] = row["solutions"][name]
        target["recovery_source"] = "phase_failure_recovery.json"
    pairs = [(r["reference"]["n_phases"], r["model"]["n_phases"]) for r in corrected["records"]]
    report["corrected_panel"] = {"total": len(pairs), "agreement_count": sum(a == b for a, b in pairs),
        "jointly_converged": sum(r["reference"]["success"] and r["model"]["success"] for r in corrected["records"]),
        "confusion_matrix_rows_reference_1_2": [[sum(a == i and b == j for a, b in pairs) for j in (1, 2)] for i in (1, 2)],
        "reference_one_phase": sum(a == 1 for a, b in pairs), "reference_two_phase": sum(a == 2 for a, b in pairs)}
    report["protocol"] = {"allocation_bounds": [1e-10, 1-1e-10], "deterministic_starts": 24,
        "random_starts": 32, "seed": 947, "mu_tolerance": 1e-8, "mass_tolerance": 1e-12,
        "tpd_tolerance": 1e-8, "unchanged_other_records": len(pairs)-len(report["recovered_records"])}
    (ROOT / "results/phase_failure_recovery.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (ROOT / "results/ternary_phase_classification_corrected.json").write_text(json.dumps(corrected, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["corrected_panel"], indent=2), flush=True)


if __name__ == "__main__":
    main()
