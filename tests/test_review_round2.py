"""Encoder-interface unit tests and a transparent audit of archived phase labels.

The graphs below are abstract fixtures, not molecules. No encoder is trained
and no molecular generalization result is inferred from these tests.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thermo import (  # noqa: E402
    PairPotentialSpec,
    initialize_pair_potential,
    ln_gamma_from_potential,
    pair_potential_g_grad,
    pair_potential_predict,
)


def shared_graph_encoder(adjacency, atom_features, weights):
    """One shared message-passing step and a permutation-invariant sum readout."""
    self_weight, neighbor_weight = weights
    hidden = np.tanh(atom_features @ self_weight + adjacency @ atom_features @ neighbor_weight)
    return hidden.sum(axis=0)


def fixtures():
    rng = np.random.default_rng(20260923)
    weights = (rng.normal(size=(4, 5)), rng.normal(size=(4, 5)))
    graphs = []
    for count in (3, 4, 6):
        adjacency = np.zeros((count, count))
        for i in range(count - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
        graphs.append((adjacency, rng.normal(size=(count, 4))))
    embeddings = np.stack([shared_graph_encoder(a, x, weights) for a, x in graphs])
    spec = PairPotentialSpec(embeddings, hidden_units=7)
    parameters = initialize_pair_potential(spec, seed=237)
    x = rng.dirichlet(np.full(3, 1.3), size=40)
    temperature = rng.uniform(300.0, 430.0, len(x))
    return rng, graphs, weights, spec, parameters, x, temperature


def test_atom_relabeling():
    rng, graphs, weights, _, _, _, _ = fixtures()
    residual = 0.0
    for adjacency, features in graphs:
        original = shared_graph_encoder(adjacency, features, weights)
        for _ in range(12):
            order = rng.permutation(len(features))
            permuted = shared_graph_encoder(adjacency[np.ix_(order, order)], features[order], weights)
            residual = max(residual, float(np.max(np.abs(original - permuted))))
    assert residual < 2.0e-12
    return {"max_embedding_residual": residual, "atom_relabelings": 36}


def test_all_species_permutations():
    _, _, _, spec, parameters, x, temperature = fixtures()
    g, gradient, activity = pair_potential_predict(parameters, spec, x, temperature)
    residuals = {"energy": 0.0, "gradient": 0.0, "activity": 0.0}
    for order_tuple in permutations(range(3)):
        order = np.asarray(order_tuple)
        other = PairPotentialSpec(spec.embeddings[order], hidden_units=spec.hidden_units)
        gp, dp, ap = pair_potential_predict(parameters, other, x[:, order], temperature)
        for key, delta in (("energy", gp - g), ("gradient", dp - gradient[:, order]), ("activity", ap - activity[:, order])):
            residuals[key] = max(residuals[key], float(np.max(np.abs(delta))))
    assert max(residuals.values()) < 2.0e-12
    return {"permutations": 6, "states_per_permutation": 40, "max_residuals": residuals}


def test_dense_embedding_pure_limits():
    _, _, _, spec, parameters, _, _ = fixtures()
    g, gradient = pair_potential_g_grad(parameters, spec, np.eye(3), np.full(3, 323.15))
    activity = ln_gamma_from_potential(g, gradient, np.eye(3))
    energy = float(np.max(np.abs(g)))
    resident = float(np.max(np.abs(np.diag(activity))))
    assert energy < 1.0e-13 and resident < 1.0e-13
    return {"max_pure_energy": energy, "max_resident_ln_gamma": resident}


def test_dense_embedding_composition_derivatives():
    _, _, _, spec, parameters, x, temperature = fixtures()
    _, derivative, _ = pair_potential_predict(parameters, spec, x, temperature)
    epsilon = 1.0e-6
    residual = 0.0
    for coordinate in range(3):
        offset = np.zeros_like(x)
        offset[:, coordinate] = epsilon
        plus = pair_potential_g_grad(parameters, spec, x + offset, temperature)[0]
        minus = pair_potential_g_grad(parameters, spec, x - offset, temperature)[0]
        residual = max(residual, float(np.max(np.abs((plus - minus) / (2 * epsilon) - derivative[:, coordinate]))))
    assert residual < 2.0e-8
    return {"centered_step": epsilon, "max_ambient_gradient_residual": residual}


def audit_phase_records():
    records = json.loads((ROOT / "results/ternary_phase_classification.json").read_text())["records"]
    jointly_converged = [r for r in records if r["reference"]["success"] and r["model"]["success"]]
    agrees = lambda r: r["reference"]["n_phases"] == r["model"]["n_phases"]
    unresolved = [i for i, r in enumerate(records) if not (r["reference"]["success"] and r["model"]["success"])]
    return {
        "total_records": len(records),
        "all_feed_label_agreements_including_fallbacks": sum(map(agrees, records)),
        "jointly_converged_records": len(jointly_converged),
        "jointly_converged_agreements": sum(map(agrees, jointly_converged)),
        "unresolved_zero_based_indices": unresolved,
        "interpretation": "Agreement with numerical reference only; no continuum global certificate. Failed-solver single-feed fallbacks are unresolved.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {
        "scope": "Four interface unit tests on untrained abstract graphs; no molecular training or chemical transfer experiment.",
        "atom_relabeling": test_atom_relabeling(),
        "species_relabeling": test_all_species_permutations(),
        "pure_limits": test_dense_embedding_pure_limits(),
        "composition_derivative": test_dense_embedding_composition_derivatives(),
        "archived_phase_convergence_audit": audit_phase_records(),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print("4 encoder-interface tests passed; archived convergence audit completed.")


if __name__ == "__main__":
    main()
