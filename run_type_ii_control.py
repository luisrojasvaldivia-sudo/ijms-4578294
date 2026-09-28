#!/usr/bin/env python3
"""Supplemental synthetic type-II ternary topology and seed-stability audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_all import COLORS, composition_samples_ternary, nrtl_predict, order_phases, save_figure  # noqa: E402
from experiments.run_major_revision import clopper_pearson, median_iqr  # noqa: E402
from src.thermo import PairPotentialSpec, canonical_embeddings, fit_pair_potential, flash_two_phase, ideal_mixing_energy, pair_potential_predict  # noqa: E402

RESULTS = ROOT / "results"
SECTIONS = ROOT / "sections"


def type_ii_parameters():
    # Components 1 and 2 are mutually miscible; pairs 1--3 and 2--3 are
    # partially miscible, producing two binary-edge miscibility gaps.
    tau_constant = np.array([[0.0, 0.25, 2.0], [0.25, 0.0, 2.0], [2.0, 2.0, 0.0]])
    tau_inverse = np.zeros((3, 3))
    alpha = np.full((3, 3), 0.20)
    np.fill_diagonal(alpha, 0.0)
    return tau_constant, tau_inverse, alpha


def make_g(reference_parameters, spec=None, learned=None):
    def evaluator(x):
        xx = np.asarray(x, dtype=float)[None, :]
        if learned is None:
            excess = nrtl_predict(xx, np.array([323.15]), reference_parameters)[0][0]
        else:
            excess = pair_potential_predict(learned, spec, xx, np.array([323.15]))[0][0]
        return float(ideal_mixing_energy(xx)[0] + excess)

    return evaluator


def main():
    reference_parameters = type_ii_parameters()
    x_train = composition_samples_ternary(np.random.default_rng(18000), 720)
    x_test = composition_samples_ternary(np.random.default_rng(18001), 1800)
    temperature_train = np.full(len(x_train), 323.15)
    temperature_test = np.full(len(x_test), 323.15)
    clean = nrtl_predict(x_train, temperature_train, reference_parameters)[2]
    target = nrtl_predict(x_test, temperature_test, reference_parameters)[2]
    spec = PairPotentialSpec(canonical_embeddings(3), hidden_units=20, temperature_center=323.15, temperature_scale=50.0)
    learned_models = []
    fit_rows = []
    for seed in range(10):
        noisy = clean + np.random.default_rng(18100 + seed).normal(scale=0.006, size=clean.shape)
        fit = fit_pair_potential(spec, x_train, temperature_train, noisy, seed=18200 + seed, maxiter=800, l2_weight=3.0e-8)
        prediction = pair_potential_predict(fit.parameters, spec, x_test, temperature_test)[2]
        fit_rows.append({"seed": seed, "test_mae": float(np.mean(np.abs(prediction - target))), "converged": fit.converged})
        learned_models.append(fit.parameters)

    reference_g = make_g(reference_parameters)
    model_g = make_g(reference_parameters, spec, learned_models[0])
    rng = np.random.default_rng(18300)
    candidates = np.vstack(
        [
            rng.dirichlet([0.35, 0.35, 0.35], 150),
            rng.dirichlet([3.0, 3.0, 3.0], 100),
            rng.dirichlet([10.0, 1.0, 1.0], 50),
            rng.dirichlet([1.0, 10.0, 1.0], 50),
            rng.dirichlet([1.0, 1.0, 10.0], 50),
        ]
    )
    prescreen = [flash_two_phase(feed, reference_g, seed=18400 + index, starts=1, ftol=1.0e-8, maxiter=250) for index, feed in enumerate(candidates)]
    provisional_single = [index for index, result in enumerate(prescreen) if result["n_phases"] == 1][:85]
    provisional_two = [index for index, result in enumerate(prescreen) if result["n_phases"] == 2][:85]
    rigorous_indices = provisional_single + provisional_two
    rigorous = [order_phases(flash_two_phase(candidates[index], reference_g, seed=19000 + position, starts=6, ftol=1.0e-10, maxiter=500)) for position, index in enumerate(rigorous_indices)]
    rigorous_single = [index for index, result in zip(rigorous_indices, rigorous) if result["n_phases"] == 1][:50]
    rigorous_two = [index for index, result in zip(rigorous_indices, rigorous) if result["n_phases"] == 2][:50]
    if len(rigorous_single) < 50 or len(rigorous_two) < 50:
        raise RuntimeError("type-II reference did not yield a balanced 100-feed panel")
    selected_indices = rigorous_single + rigorous_two
    feeds = candidates[selected_indices]
    reference_results = [order_phases(flash_two_phase(feed, reference_g, seed=20000 + index, starts=8, ftol=1.0e-11, maxiter=600)) for index, feed in enumerate(feeds)]
    model_results = [order_phases(flash_two_phase(feed, model_g, seed=21000 + index, starts=8, ftol=1.0e-11, maxiter=600)) for index, feed in enumerate(feeds)]
    reference_labels = np.array([result["n_phases"] == 2 for result in reference_results], dtype=int)
    model_labels = np.array([result["n_phases"] == 2 for result in model_results], dtype=int)
    matrix = confusion_matrix(reference_labels, model_labels, labels=[0, 1])
    correct = int(np.sum(reference_labels == model_labels))

    # Twenty smallest opposite-class distances form the boundary stress panel.
    distance = np.empty(len(feeds))
    for index, feed in enumerate(feeds):
        distance[index] = np.min(np.linalg.norm(feeds[reference_labels != reference_labels[index]] - feed, axis=1))
    stress_indices = np.argsort(distance)[:20]
    seed_phase_accuracy = []
    for seed, parameters in enumerate(learned_models):
        seed_model_g = make_g(reference_parameters, spec, parameters)
        labels = []
        for local_index, state_index in enumerate(stress_indices):
            result = flash_two_phase(feeds[state_index], seed_model_g, seed=22000 + 100 * seed + local_index, starts=6, ftol=1.0e-10, maxiter=500)
            labels.append(result["n_phases"] == 2)
        seed_phase_accuracy.append(float(np.mean(np.asarray(labels, dtype=int) == reference_labels[stress_indices])))

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    axes[0].imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=11)
    axes[0].set_xticks([0, 1], ["One", "Two"])
    axes[0].set_yticks([0, 1], ["One", "Two"])
    axes[0].set_xlabel("Predicted phases")
    axes[0].set_ylabel("Reference phases")
    axes[0].set_title("(a) Type-II 100-feed panel")
    axes[1].scatter(np.arange(1, 11), seed_phase_accuracy, color=COLORS["potential"])
    axes[1].axhline(np.median(seed_phase_accuracy), color=COLORS["reference"], linestyle="--", label="Median")
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Boundary-panel accuracy")
    axes[1].set_ylim(0, 1.04)
    axes[1].set_title("(b) Twenty-feed boundary stress")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig11_type_ii_control")

    payload = {
        "epistemic_label": "E7 synthetic type-II topology",
        "nrtl_parameters": {"tau_constant": reference_parameters[0].tolist(), "alpha": reference_parameters[2].tolist()},
        "fit_replicates": fit_rows,
        "fit_test_mae": median_iqr([row["test_mae"] for row in fit_rows]),
        "feed_count": int(len(feeds)),
        "reference_one_phase": int(np.sum(reference_labels == 0)),
        "reference_two_phase": int(np.sum(reference_labels == 1)),
        "confusion_matrix_rows_reference": matrix.tolist(),
        "phase_accuracy": correct / len(feeds),
        "phase_accuracy_clopper_pearson_95": clopper_pearson(correct, len(feeds)),
        "boundary_stress_seed_accuracy": median_iqr(seed_phase_accuracy),
        "boundary_stress_indices": stress_indices.tolist(),
    }
    exponent = int(np.floor(np.log10(payload["fit_test_mae"]["median"])))
    mantissa = payload["fit_test_mae"]["median"] / 10**exponent
    (RESULTS / "type_ii_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (SECTIONS / "type_ii_macros.tex").write_text(
        "% Auto-generated by experiments/run_type_ii_control.py.\n"
        + rf"\newcommand{{\TypeIIMAE}}{{${mantissa:.2f}\times 10^{{{exponent}}}$}}" + "\n"
        + rf"\newcommand{{\TypeIIAccuracy}}{{{100 * payload['phase_accuracy']:.1f}\%}}" + "\n"
        + rf"\newcommand{{\TypeIIAccuracyLow}}{{{100 * payload['phase_accuracy_clopper_pearson_95'][0]:.1f}\%}}" + "\n"
        + rf"\newcommand{{\TypeIIAccuracyHigh}}{{{100 * payload['phase_accuracy_clopper_pearson_95'][1]:.1f}\%}}" + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in ("fit_test_mae", "phase_accuracy", "phase_accuracy_clopper_pearson_95", "boundary_stress_seed_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
