#!/usr/bin/env python3
"""Execute the full synthetic verification campaign and regenerate all figures.

Every numerical value imported by the LaTeX manuscript is written to
``sections/legacy_results_macros.tex`` from the archived JSON results.  The script is
CPU-only and deterministic under the declared seeds.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import autograd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import spearmanr, wilcoxon
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thermo import (  # noqa: E402
    PairPotentialSpec,
    canonical_embeddings,
    fit_pair_potential,
    flash_two_phase,
    ideal_mixing_energy,
    lower_convex_hull_binodal,
    minimum_tpd,
    nrtl_g_grad_lngamma,
    pair_potential_g_grad,
    pair_potential_predict,
)


FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "synthetic"
SECTIONS = ROOT / "sections"
for directory in (FIGURES, RESULTS, DATA, SECTIONS):
    directory.mkdir(parents=True, exist_ok=True)

COLORS = {
    "reference": "#1f4e79",
    "potential": "#d1495b",
    "direct": "#2a9d8f",
    "uncertainty": "#7b2cbf",
    "neutral": "#5f6b73",
    "accent": "#e9c46a",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png")
    plt.close(fig)


def binary_parameters():
    tau_constant = np.zeros((2, 2))
    tau_inverse_temperature = np.array([[0.0, 850.0], [850.0, 0.0]])
    alpha = np.array([[0.0, 0.20], [0.20, 0.0]])
    return tau_constant, tau_inverse_temperature, alpha


def ternary_parameters():
    # Synthetic type-I extraction topology.  Values are model parameters, not
    # fitted claims about water, solute, or an ionic-liquid extractant.
    tau_constant = np.array(
        [
            [0.0, 0.45, 3.15],
            [0.30, 0.0, 0.55],
            [3.05, 0.70, 0.0],
        ]
    )
    tau_inverse_temperature = np.zeros((3, 3))
    alpha = np.full((3, 3), 0.20)
    np.fill_diagonal(alpha, 0.0)
    return tau_constant, tau_inverse_temperature, alpha


def nrtl_predict(x, temperature, parameters):
    return nrtl_g_grad_lngamma(x, temperature, *parameters)


def composition_samples_binary(rng, count, t_low, t_high):
    x1 = rng.uniform(0.004, 0.996, size=count)
    x = np.column_stack([x1, 1.0 - x1])
    t = rng.uniform(t_low, t_high, size=count)
    return x, t


def composition_samples_ternary(rng, count):
    return rng.dirichlet(np.full(3, 0.85), size=count)


def direct_features(x, temperature, center=370.0, scale=80.0):
    return np.column_stack([x, (np.asarray(temperature) - center) / scale])


def fit_binary_models():
    rng = np.random.default_rng(42)
    parameters = binary_parameters()
    x_train, t_train = composition_samples_binary(rng, 420, 290.0, 450.0)
    _, _, target_clean = nrtl_predict(x_train, t_train, parameters)
    target_noisy = target_clean + rng.normal(scale=0.01, size=target_clean.shape)

    x_test, t_test = composition_samples_binary(np.random.default_rng(99), 2400, 290.0, 450.0)
    g_test, _, target_test = nrtl_predict(x_test, t_test, parameters)

    spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=16)
    fit = fit_pair_potential(
        spec,
        x_train,
        t_train,
        target_noisy,
        seed=0,
        maxiter=650,
        l2_weight=3.0e-8,
    )
    g_pred, _, pred_potential = pair_potential_predict(fit.parameters, spec, x_test, t_test)

    direct = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(24, 24),
            activation="tanh",
            solver="lbfgs",
            alpha=1.0e-6,
            max_iter=1800,
            random_state=0,
            tol=1.0e-10,
        ),
    )
    started = perf_counter()
    direct.fit(direct_features(x_train, t_train), target_noisy)
    direct_elapsed = perf_counter() - started
    pred_direct = direct.predict(direct_features(x_test, t_test))

    np.savez_compressed(
        DATA / "binary_train_test.npz",
        x_train=x_train,
        temperature_train=t_train,
        ln_gamma_train=target_noisy,
        x_test=x_test,
        temperature_test=t_test,
        ln_gamma_test=target_test,
    )
    np.savez_compressed(
        RESULTS / "binary_model.npz",
        parameters=fit.parameters,
        embeddings=spec.embeddings,
        hidden_units=spec.hidden_units,
    )

    potential_state_mae = np.mean(np.abs(pred_potential - target_test), axis=1)
    direct_state_mae = np.mean(np.abs(pred_direct - target_test), axis=1)
    paired_improvement = direct_state_mae - potential_state_mae
    bootstrap_rng = np.random.default_rng(2026)
    bootstrap_means = np.empty(2000)
    for index in range(bootstrap_means.size):
        selection = bootstrap_rng.integers(0, paired_improvement.size, size=paired_improvement.size)
        bootstrap_means[index] = float(np.mean(paired_improvement[selection]))
    wilcoxon_result = wilcoxon(
        direct_state_mae,
        potential_state_mae,
        alternative="greater",
        method="approx",
    )

    metrics = {
        "potential_mae": float(np.mean(np.abs(pred_potential - target_test))),
        "direct_mae": float(np.mean(np.abs(pred_direct - target_test))),
        "potential_g_mae": float(np.mean(np.abs(g_pred - g_test))),
        "potential_train_seconds": fit.elapsed_seconds,
        "direct_train_seconds": float(direct_elapsed),
        "potential_iterations": fit.iterations,
        "potential_converged": fit.converged,
        "potential_message": fit.message,
        "parameter_count_potential": spec.parameter_count,
        "parameter_count_direct": int(sum(layer.size for layer in direct[-1].coefs_) + sum(layer.size for layer in direct[-1].intercepts_)),
        "mae_improvement_factor": float(np.mean(direct_state_mae) / np.mean(potential_state_mae)),
        "paired_mae_difference": float(np.mean(paired_improvement)),
        "paired_bootstrap_ci95": [float(np.quantile(bootstrap_means, 0.025)), float(np.quantile(bootstrap_means, 0.975))],
        "wilcoxon_statistic": float(wilcoxon_result.statistic),
        "wilcoxon_pvalue": float(wilcoxon_result.pvalue),
    }
    return parameters, spec, fit.parameters, direct, metrics, (x_test, t_test, g_test, target_test, g_pred, pred_potential, pred_direct)


def gibbs_duhem_diagnostics(binary_params, spec, learned_parameters, direct):
    x1 = np.linspace(0.003, 0.997, 1201)
    x = np.column_stack([x1, 1.0 - x1])
    temperatures = np.array([300.0, 350.0, 400.0, 440.0])
    rows = []
    for temperature in temperatures:
        t = np.full(x.shape[0], temperature)
        _, _, ref = nrtl_predict(x, t, binary_params)
        _, _, pot = pair_potential_predict(learned_parameters, spec, x, t)
        direct_pred = direct.predict(direct_features(x, t))
        row = {"temperature": temperature}
        for name, values in (("reference", ref), ("potential", pot), ("direct", direct_pred)):
            derivative = np.gradient(values, x1, axis=0, edge_order=2)
            gd = np.abs(np.sum(x * derivative, axis=1))
            row[name] = float(np.mean(gd[8:-8]))
        rows.append(row)

    # Exact pure-component boundary is evaluated through the unconstrained core.
    pure = np.eye(2)
    g_pure, grad_pure = pair_potential_g_grad(
        learned_parameters, spec, pure, np.array([350.0, 350.0])
    )
    weighted = np.sum(pure * np.asarray(grad_pure), axis=1, keepdims=True)
    lng_pure = np.asarray(g_pure)[:, None] + np.asarray(grad_pure) - weighted
    resident_pure_residual = float(max(abs(lng_pure[0, 0]), abs(lng_pure[1, 1])))
    direct_pure = direct.predict(direct_features(pure, np.array([350.0, 350.0])))
    direct_pure_residual = float(max(abs(direct_pure[0, 0]), abs(direct_pure[1, 1])))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    positions = np.arange(len(temperatures))
    width = 0.24
    for offset, name, color, label in (
        (-width, "reference", COLORS["reference"], "Analytic NRTL floor"),
        (0.0, "potential", COLORS["potential"], "Constrained potential"),
        (width, "direct", COLORS["direct"], "Direct activity model"),
    ):
        ax.bar(positions + offset, [r[name] for r in rows], width=width, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(positions, [f"{int(t)} K" for t in temperatures])
    ax.set_ylabel("Mean Gibbs-Duhem residual")
    ax.set_title("Differential consistency across temperature")
    ax.grid(axis="y", which="both", alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout()
    save_figure(fig, "fig03_gibbs_duhem")

    return {
        "rows": rows,
        "potential_mean": float(np.mean([r["potential"] for r in rows])),
        "direct_mean": float(np.mean([r["direct"] for r in rows])),
        "reference_mean": float(np.mean([r["reference"] for r in rows])),
        "potential_pure_residual": resident_pure_residual,
        "direct_pure_residual": direct_pure_residual,
    }


def plot_framework():
    fig, ax = plt.subplots(figsize=(8.0, 3.25))
    ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.18, 0.26, "Species identities\nComposition and T", COLORS["reference"]),
        (0.27, 0.58, 0.18, 0.26, "Symmetric pair\nfree-energy model", COLORS["potential"]),
        (0.52, 0.58, 0.18, 0.26, "Thermodynamic\nderivatives", COLORS["uncertainty"]),
        (0.77, 0.58, 0.20, 0.26, "Convex hull and\nTPD stability", COLORS["direct"]),
        (0.37, 0.10, 0.26, 0.24, "Uncertainty-aware flash\nand selective abstention", COLORS["neutral"]),
    ]
    for x0, y0, width, height, label, color in boxes:
        patch = plt.Rectangle((x0, y0), width, height, transform=ax.transAxes, facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x0 + width / 2, y0 + height / 2, label, transform=ax.transAxes, ha="center", va="center", color="#202a33", fontsize=8.2)
    arrow = dict(arrowstyle="-|>", lw=1.25, color="#56616b")
    for start, end in ((0.20, 0.27), (0.45, 0.52), (0.70, 0.77)):
        ax.annotate("", xy=(end, 0.71), xytext=(start, 0.71), xycoords=ax.transAxes, arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.34), xytext=(0.62, 0.58), xycoords=ax.transAxes, arrowprops=arrow)
    ax.annotate("", xy=(0.76, 0.57), xytext=(0.63, 0.29), xycoords=ax.transAxes, arrowprops=arrow)
    ax.text(0.5, 0.96, "Thermodynamic structure is enforced before equilibrium is solved", transform=ax.transAxes, ha="center", va="center", weight="bold", fontsize=10.5)
    save_figure(fig, "fig01_framework")


def plot_binary_fit(binary_params, spec, learned_parameters, test_payload):
    x_test, t_test, _, target_test, _, pred_potential, pred_direct = test_payload
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    x1 = np.linspace(0.003, 0.997, 600)
    x = np.column_stack([x1, 1.0 - x1])
    for temperature, line_style in ((300.0, "-"), (360.0, "--"), (430.0, ":")):
        t = np.full(x.shape[0], temperature)
        g_ref, _, _ = nrtl_predict(x, t, binary_params)
        g_model, _, _ = pair_potential_predict(learned_parameters, spec, x, t)
        axes[0].plot(x1, g_ref, color=COLORS["reference"], linestyle=line_style, linewidth=1.5)
        axes[0].plot(x1, g_model, color=COLORS["potential"], linestyle=line_style, linewidth=1.1)
    axes[0].set_xlabel(r"Mole fraction $x_1$")
    axes[0].set_ylabel(r"$g^{E}/(RT)$")
    axes[0].set_title("(a) Excess free-energy recovery")
    axes[0].text(0.03, 0.96, "Solid: 300 K | dashed: 360 K | dotted: 430 K", transform=axes[0].transAxes, va="top", fontsize=7.2)

    sample = np.random.default_rng(123).choice(len(x_test), size=900, replace=False)
    axes[1].scatter(target_test[sample].ravel(), pred_direct[sample].ravel(), s=6, alpha=0.28, color=COLORS["direct"], label="Direct model")
    axes[1].scatter(target_test[sample].ravel(), pred_potential[sample].ravel(), s=6, alpha=0.35, color=COLORS["potential"], label="Constrained potential")
    limits = [min(target_test.min(), pred_direct.min(), pred_potential.min()), max(target_test.max(), pred_direct.max(), pred_potential.max())]
    axes[1].plot(limits, limits, color="#222222", linewidth=0.8)
    axes[1].set_xlim(limits)
    axes[1].set_ylim(limits)
    axes[1].set_xlabel(r"Reference $\ln\gamma_i$")
    axes[1].set_ylabel(r"Predicted $\ln\gamma_i$")
    axes[1].set_title("(b) Held-out parity")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "fig02_binary_fit")


def binary_binodal_experiment(binary_params, spec, learned_parameters):
    temperatures = np.linspace(292.0, 448.0, 40)
    x1_grid = np.linspace(0.001, 0.999, 2600)
    x_grid = np.column_stack([x1_grid, 1.0 - x1_grid])
    reference, predicted = [], []
    for temperature in temperatures:
        t = np.full(x_grid.shape[0], temperature)
        g_ref, _, _ = nrtl_predict(x_grid, t, binary_params)
        g_pred, _, _ = pair_potential_predict(learned_parameters, spec, x_grid, t)
        binodal_ref = lower_convex_hull_binodal(x1_grid, ideal_mixing_energy(x_grid) + g_ref)
        binodal_pred = lower_convex_hull_binodal(x1_grid, ideal_mixing_energy(x_grid) + g_pred)
        if binodal_ref is not None and binodal_pred is not None:
            reference.append([temperature, *binodal_ref])
            predicted.append([temperature, *binodal_pred])
    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    if reference.shape != predicted.shape or len(reference) < 10:
        raise RuntimeError("binodal recovery failed for too many temperatures")

    t_min, t_max = temperatures.min(), temperatures.max()
    ref_points = np.vstack(
        [
            np.column_stack([reference[:, 1], (reference[:, 0] - t_min) / (t_max - t_min)]),
            np.column_stack([reference[:, 2], (reference[:, 0] - t_min) / (t_max - t_min)]),
        ]
    )
    pred_points = np.vstack(
        [
            np.column_stack([predicted[:, 1], (predicted[:, 0] - t_min) / (t_max - t_min)]),
            np.column_stack([predicted[:, 2], (predicted[:, 0] - t_min) / (t_max - t_min)]),
        ]
    )
    hausdorff = max(directed_hausdorff(ref_points, pred_points)[0], directed_hausdorff(pred_points, ref_points)[0])
    composition_mae = float(np.mean(np.abs(reference[:, 1:] - predicted[:, 1:])))

    fig, ax = plt.subplots(figsize=(4.9, 4.1))
    ax.plot(reference[:, 1], reference[:, 0], color=COLORS["reference"], linewidth=1.8, label="NRTL reference")
    ax.plot(reference[:, 2], reference[:, 0], color=COLORS["reference"], linewidth=1.8)
    ax.plot(predicted[:, 1], predicted[:, 0], "--", color=COLORS["potential"], linewidth=1.5, label="Constrained potential")
    ax.plot(predicted[:, 2], predicted[:, 0], "--", color=COLORS["potential"], linewidth=1.5)
    ax.set_xlabel(r"Mole fraction $x_1$")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Binary liquid-liquid binodal")
    ax.legend(frameon=False, loc="lower center")
    ax.grid(alpha=0.16)
    fig.tight_layout()
    save_figure(fig, "fig04_binary_binodal")

    np.savetxt(DATA / "binary_binodal_reference.csv", reference, delimiter=",", header="temperature_K,x1_left,x1_right", comments="")
    np.savetxt(DATA / "binary_binodal_prediction.csv", predicted, delimiter=",", header="temperature_K,x1_left,x1_right", comments="")
    return {"hausdorff_scaled": float(hausdorff), "composition_mae": composition_mae, "isotherms": int(len(reference))}


def fit_ternary_model():
    rng = np.random.default_rng(7)
    parameters = ternary_parameters()
    x_train = composition_samples_ternary(rng, 720)
    t_train = np.full(x_train.shape[0], 323.15)
    _, _, target = nrtl_predict(x_train, t_train, parameters)
    noisy = target + rng.normal(scale=0.006, size=target.shape)
    spec = PairPotentialSpec(canonical_embeddings(3), hidden_units=20, temperature_center=323.15, temperature_scale=50.0)
    fit = fit_pair_potential(spec, x_train, t_train, noisy, seed=3, maxiter=850, l2_weight=3.0e-8)
    x_test = composition_samples_ternary(np.random.default_rng(71), 1800)
    t_test = np.full(x_test.shape[0], 323.15)
    g_ref, _, y_ref = nrtl_predict(x_test, t_test, parameters)
    g_pred, _, y_pred = pair_potential_predict(fit.parameters, spec, x_test, t_test)
    np.savez_compressed(DATA / "ternary_train_test.npz", x_train=x_train, temperature_train=t_train, ln_gamma_train=noisy, x_test=x_test, ln_gamma_test=y_ref)
    np.savez_compressed(RESULTS / "ternary_model.npz", parameters=fit.parameters, embeddings=spec.embeddings, hidden_units=spec.hidden_units)
    metrics = {
        "ln_gamma_mae": float(np.mean(np.abs(y_pred - y_ref))),
        "g_excess_mae": float(np.mean(np.abs(g_pred - g_ref))),
        "train_seconds": fit.elapsed_seconds,
        "iterations": fit.iterations,
        "converged": fit.converged,
        "message": fit.message,
    }
    return parameters, spec, fit.parameters, metrics


def make_g_functions(nrtl_params, spec, learned_parameters, temperature=323.15):
    def reference_g(x):
        xx = np.asarray(x, dtype=float)[None, :]
        g, _, _ = nrtl_predict(xx, np.array([temperature]), nrtl_params)
        return float(ideal_mixing_energy(xx)[0] + g[0])

    def model_g(x):
        xx = np.asarray(x, dtype=float)[None, :]
        g, _, _ = pair_potential_predict(learned_parameters, spec, xx, np.array([temperature]))
        return float(ideal_mixing_energy(xx)[0] + g[0])

    def reference_lng(x):
        xx = np.asarray(x, dtype=float)[None, :]
        return nrtl_predict(xx, np.array([temperature]), nrtl_params)[2][0]

    def model_lng(x):
        xx = np.asarray(x, dtype=float)[None, :]
        return pair_potential_predict(learned_parameters, spec, xx, np.array([temperature]))[2][0]

    return reference_g, model_g, reference_lng, model_lng


def order_phases(result):
    result = dict(result)
    if result["phase_a"][0] < result["phase_b"][0]:
        result["phase_a"], result["phase_b"] = result["phase_b"], result["phase_a"]
        result["beta"] = 1.0 - result["beta"]
    return result


def ternary_flash_experiment(nrtl_params, spec, learned_parameters):
    reference_g, model_g, reference_lng, model_lng = make_g_functions(nrtl_params, spec, learned_parameters)
    feeds = np.array(
        [
            [0.46, 0.10, 0.44],
            [0.54, 0.11, 0.35],
            [0.36, 0.15, 0.49],
            [0.50, 0.18, 0.32],
            [0.31, 0.19, 0.50],
            [0.59, 0.15, 0.26],
        ]
    )
    records = []
    for idx, feed in enumerate(feeds):
        reference = order_phases(flash_two_phase(feed, reference_g, seed=100 + idx, starts=12))
        model = order_phases(flash_two_phase(feed, model_g, seed=200 + idx, starts=12))
        tpd_values = []
        if model["n_phases"] == 2:
            tpd_values.append(minimum_tpd(model["phase_a"], model_lng, seed=300 + idx, starts=18))
            tpd_values.append(minimum_tpd(model["phase_b"], model_lng, seed=400 + idx, starts=18))
        records.append(
            {
                "feed": feed.tolist(),
                "reference": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in reference.items() if k != "success"},
                "model": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items() if k != "success"},
                "tpd": tpd_values,
            }
        )

    phase_accuracy = float(np.mean([r["reference"]["n_phases"] == r["model"]["n_phases"] for r in records]))
    two_phase_records = [r for r in records if r["reference"]["n_phases"] == 2 and r["model"]["n_phases"] == 2]
    if not two_phase_records:
        raise RuntimeError("ternary system did not generate matched two-phase states")
    tie_errors, beta_errors, tpd_all = [], [], []
    for record in two_phase_records:
        ref_a = np.asarray(record["reference"]["phase_a"])
        ref_b = np.asarray(record["reference"]["phase_b"])
        mod_a = np.asarray(record["model"]["phase_a"])
        mod_b = np.asarray(record["model"]["phase_b"])
        tie_errors.extend(np.abs(mod_a - ref_a).tolist())
        tie_errors.extend(np.abs(mod_b - ref_b).tolist())
        beta_errors.append(abs(record["model"]["beta"] - record["reference"]["beta"]))
        tpd_all.extend(record["tpd"])

    # Barycentric projection for a publication-quality ternary plot.
    def project(points):
        points = np.asarray(points)
        return np.column_stack([points[:, 2] + 0.5 * points[:, 1], (np.sqrt(3.0) / 2.0) * points[:, 1]])

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    vertices = project(np.eye(3))
    outline = np.vstack([vertices, vertices[0]])
    ax.plot(outline[:, 0], outline[:, 1], color="#2f3640", linewidth=1.2)
    for record in records:
        feed_xy = project(np.asarray(record["feed"])[None, :])[0]
        ax.scatter(*feed_xy, color=COLORS["accent"], edgecolor="#8a6d1d", s=30, zorder=4)
        for key, color, style in (("reference", COLORS["reference"], "-"), ("model", COLORS["potential"], "--")):
            if record[key]["n_phases"] == 2:
                endpoints = project(np.vstack([record[key]["phase_a"], record[key]["phase_b"]]))
                ax.plot(endpoints[:, 0], endpoints[:, 1], linestyle=style, color=color, linewidth=1.3)
    ax.text(vertices[0, 0] - 0.03, vertices[0, 1] - 0.05, "Water-like (1)", ha="right")
    ax.text(vertices[1, 0], vertices[1, 1] + 0.04, "Solute (2)", ha="center")
    ax.text(vertices[2, 0] + 0.03, vertices[2, 1] - 0.05, "Extractant-like (3)", ha="left")
    ax.plot([], [], color=COLORS["reference"], label="NRTL tie-lines")
    ax.plot([], [], "--", color=COLORS["potential"], label="Learned tie-lines")
    ax.scatter([], [], color=COLORS["accent"], edgecolor="#8a6d1d", label="Feeds")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Ternary type-I liquid-liquid equilibrium")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "fig05_ternary_tielines")

    with (RESULTS / "ternary_flash_records.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    return {
        "phase_accuracy": phase_accuracy,
        "matched_two_phase": len(two_phase_records),
        "feed_count": len(records),
        "tie_line_mae": float(np.mean(tie_errors)),
        "beta_mae": float(np.mean(beta_errors)),
        "tpd_median": float(np.median(tpd_all)) if tpd_all else float("nan"),
        "tpd_worst": float(np.min(tpd_all)) if tpd_all else float("nan"),
    }


def uncertainty_experiment(binary_params):
    ensemble_parameters = []
    fits = []
    spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=12)
    for seed in range(5):
        rng = np.random.default_rng(110 + seed)
        x_train, t_train = composition_samples_binary(rng, 280, 290.0, 370.0)
        _, _, y = nrtl_predict(x_train, t_train, binary_params)
        y = y + rng.normal(scale=0.01, size=y.shape)
        fit = fit_pair_potential(spec, x_train, t_train, y, seed=20 + seed, maxiter=430, l2_weight=5.0e-8)
        ensemble_parameters.append(fit.parameters)
        fits.append(fit)

    rng = np.random.default_rng(911)
    x_in, t_in = composition_samples_binary(rng, 400, 295.0, 365.0)
    x_out, t_out = composition_samples_binary(rng, 400, 390.0, 450.0)
    x = np.vstack([x_in, x_out])
    t = np.concatenate([t_in, t_out])
    domain = np.concatenate([np.zeros(len(x_in), dtype=int), np.ones(len(x_out), dtype=int)])
    _, _, target = nrtl_predict(x, t, binary_params)
    predictions = np.stack([pair_potential_predict(p, spec, x, t)[2] for p in ensemble_parameters])
    mean_prediction = predictions.mean(axis=0)
    point_error = np.mean(np.abs(mean_prediction - target), axis=1)
    uncertainty = np.sqrt(np.mean(np.var(predictions, axis=0, ddof=1), axis=1))
    spearman_result = spearmanr(uncertainty, point_error)
    rho = float(spearman_result.statistic)
    order = np.argsort(uncertainty)[::-1]
    top_half = order[: len(order) // 2]
    capture = float(np.mean(domain[top_half] == 1) * len(top_half) / max(domain.sum(), 1))

    coverages = np.linspace(1.0, 0.20, 17)
    selective, random_curve = [], []
    rng_random = np.random.default_rng(991)
    random_scores = rng_random.random(len(x))
    for coverage in coverages:
        keep_count = max(5, int(round(coverage * len(x))))
        keep_selective = np.argsort(uncertainty)[:keep_count]
        keep_random = np.argsort(random_scores)[:keep_count]
        selective.append(float(np.mean(point_error[keep_selective])))
        random_curve.append(float(np.mean(point_error[keep_random])))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    axes[0].scatter(uncertainty[domain == 0], point_error[domain == 0], s=9, alpha=0.42, color=COLORS["reference"], label="In-domain")
    axes[0].scatter(uncertainty[domain == 1], point_error[domain == 1], s=13, alpha=0.50, marker="^", color=COLORS["potential"], label="Temperature OOD")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ensemble standard deviation")
    axes[0].set_ylabel("Mean absolute state error")
    axes[0].set_title(f"(a) Error ranking: Spearman rho = {rho:.3f}")
    axes[0].legend(frameon=False)

    referred = 100.0 * (1.0 - coverages)
    axes[1].plot(referred, selective, marker="o", color=COLORS["uncertainty"], label="Uncertainty-guided")
    axes[1].plot(referred, random_curve, marker="s", color=COLORS["neutral"], label="Random")
    axes[1].set_xlabel("States referred to rigorous solver (%)")
    axes[1].set_ylabel("MAE of accepted states")
    axes[1].set_title("(b) Selective prediction")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig06_uncertainty")

    np.savez_compressed(
        RESULTS / "uncertainty_ensemble.npz",
        parameters=np.stack(ensemble_parameters),
        x=x,
        temperature=t,
        domain=domain,
        target=target,
        prediction=mean_prediction,
        uncertainty=uncertainty,
        error=point_error,
    )
    return {
        "spearman_rho": rho,
        "spearman_pvalue": float(spearman_result.pvalue),
        "ood_capture_top_half": capture,
        "mae_in_domain": float(np.mean(point_error[domain == 0])),
        "mae_out_domain": float(np.mean(point_error[domain == 1])),
        "ood_error_ratio": float(np.mean(point_error[domain == 1]) / np.mean(point_error[domain == 0])),
        "mae_full_coverage": selective[0],
        "mae_40pct_coverage": selective[int(np.argmin(np.abs(coverages - 0.40)))],
        "fit_seconds_total": float(sum(f.elapsed_seconds for f in fits)),
        "converged_models": int(sum(f.converged for f in fits)),
        "ensemble_size": len(fits),
    }


def permutation_equivariance_test(spec, learned_parameters):
    rng = np.random.default_rng(818)
    x = rng.dirichlet(np.ones(3), size=120)
    t = np.full(len(x), 323.15)
    _, _, original = pair_potential_predict(learned_parameters, spec, x, t)
    permutation = np.array([2, 0, 1])
    permuted_spec = PairPotentialSpec(
        embeddings=spec.embeddings[permutation],
        hidden_units=spec.hidden_units,
        temperature_center=spec.temperature_center,
        temperature_scale=spec.temperature_scale,
    )
    _, _, permuted = pair_potential_predict(learned_parameters, permuted_spec, x[:, permutation], t)
    return float(np.max(np.abs(permuted - original[:, permutation])))


def cascade_experiment(nrtl_params, spec, learned_parameters):
    reference_g, model_g, _, _ = make_g_functions(nrtl_params, spec, learned_parameters)

    def run_cascade(g_function, seed_offset):
        inventory = np.array([0.85, 0.15, 1.0e-8])
        initial_solute = inventory[1]
        raffinate_profile = [inventory[1] / inventory.sum()]
        recovery_profile = [0.0]
        balance_errors = []
        stage_seconds = []
        for stage in range(1, 5):
            mixed = inventory + np.array([1.0e-8, 1.0e-8, 0.50])
            total = mixed.sum()
            feed = mixed / total
            started = perf_counter()
            flash = order_phases(flash_two_phase(feed, g_function, seed=seed_offset + stage, starts=8))
            stage_seconds.append(perf_counter() - started)
            if flash["n_phases"] == 1:
                raffinate_moles = total
                raffinate_x = feed
                reconstructed = mixed
            else:
                # order_phases guarantees phase A is water-rich.
                raffinate_moles = flash["beta"] * total
                extract_moles = (1.0 - flash["beta"]) * total
                raffinate_x = np.asarray(flash["phase_a"])
                extract_x = np.asarray(flash["phase_b"])
                reconstructed = raffinate_moles * raffinate_x + extract_moles * extract_x
            balance_errors.append(float(np.max(np.abs(reconstructed - mixed))))
            inventory = raffinate_moles * raffinate_x
            raffinate_profile.append(float(raffinate_x[1]))
            recovery_profile.append(float(1.0 - inventory[1] / initial_solute))
        return {
            "raffinate_profile": raffinate_profile,
            "recovery_profile": recovery_profile,
            "balance_errors": balance_errors,
            "stage_seconds": stage_seconds,
        }

    reference = run_cascade(reference_g, 1300)
    model = run_cascade(model_g, 2300)
    stages = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(stages, reference["raffinate_profile"], marker="o", color=COLORS["reference"], label="NRTL reference")
    axes[0].plot(stages, model["raffinate_profile"], marker="s", linestyle="--", color=COLORS["potential"], label="Constrained potential")
    axes[0].set_xlabel("Cross-current stage")
    axes[0].set_ylabel("Solute mole fraction in raffinate")
    axes[0].set_title("(a) Raffinate depletion")
    axes[0].set_xticks(stages)
    axes[0].legend(frameon=False)
    axes[1].bar([0, 1], [100.0 * reference["recovery_profile"][-1], 100.0 * model["recovery_profile"][-1]], color=[COLORS["reference"], COLORS["potential"]], width=0.62)
    axes[1].set_xticks([0, 1], ["NRTL", "Potential"])
    axes[1].set_ylabel("Solute recovery after four stages (%)")
    axes[1].set_title("(b) Process-level propagation")
    for idx, value in enumerate([reference["recovery_profile"][-1], model["recovery_profile"][-1]]):
        axes[1].text(idx, 100.0 * value + 1.0, f"{100.0 * value:.2f}%", ha="center", fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "fig07_extraction_cascade")

    with (RESULTS / "cascade.json").open("w", encoding="utf-8") as handle:
        json.dump({"reference": reference, "model": model}, handle, indent=2)
    return {
        "reference_recovery": float(reference["recovery_profile"][-1]),
        "model_recovery": float(model["recovery_profile"][-1]),
        "recovery_abs_error": float(abs(model["recovery_profile"][-1] - reference["recovery_profile"][-1])),
        "max_balance_error": float(max(reference["balance_errors"] + model["balance_errors"])),
        "reference_seconds_mean": float(np.mean(reference["stage_seconds"])),
        "model_seconds_mean": float(np.mean(model["stage_seconds"])),
        "runtime_ratio": float(np.mean(model["stage_seconds"]) / np.mean(reference["stage_seconds"])),
    }


def write_macros(metrics):
    exp1, gd, exp2, ternary_fit, exp3, exp4, structural, exp5 = (
        metrics["exp1"],
        metrics["gibbs_duhem"],
        metrics["exp2"],
        metrics["ternary_fit"],
        metrics["exp3"],
        metrics["exp4"],
        metrics["structural"],
        metrics["exp5"],
    )

    def sci(value, digits=2):
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / 10**exponent
        return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"

    lines = [
        "% Legacy outputs from experiments/run_all.py; not imported by the revised manuscript.",
        rf"\newcommand{{\PotentialMAE}}{{${sci(exp1['potential_mae'])}$}}",
        rf"\newcommand{{\DirectMAE}}{{${sci(exp1['direct_mae'])}$}}",
        rf"\newcommand{{\MAEImprovementFactor}}{{{exp1['mae_improvement_factor']:.2f}}}",
        rf"\newcommand{{\MAEDifference}}{{${sci(exp1['paired_mae_difference'])}$}}",
        rf"\newcommand{{\MAEDifferenceCILow}}{{${sci(exp1['paired_bootstrap_ci95'][0])}$}}",
        rf"\newcommand{{\MAEDifferenceCIHigh}}{{${sci(exp1['paired_bootstrap_ci95'][1])}$}}",
        rf"\newcommand{{\WilcoxonP}}{{${sci(exp1['wilcoxon_pvalue'])}$}}",
        rf"\newcommand{{\PotentialGD}}{{${sci(gd['potential_mean'])}$}}",
        rf"\newcommand{{\DirectGD}}{{${sci(gd['direct_mean'])}$}}",
        rf"\newcommand{{\ReferenceGDFloor}}{{${sci(gd['reference_mean'])}$}}",
        rf"\newcommand{{\DirectPureResidual}}{{${sci(gd['direct_pure_residual'])}$}}",
        rf"\newcommand{{\BinaryHausdorff}}{{${sci(exp2['hausdorff_scaled'])}$}}",
        rf"\newcommand{{\BinaryCompositionMAE}}{{${sci(exp2['composition_mae'])}$}}",
        rf"\newcommand{{\TernaryGammaMAE}}{{${sci(ternary_fit['ln_gamma_mae'])}$}}",
        rf"\newcommand{{\PhaseAccuracy}}{{{int(round(100 * exp3['phase_accuracy']))}\%}}",
        rf"\newcommand{{\MatchedFeeds}}{{{exp3['matched_two_phase']}/{exp3['feed_count']}}}",
        rf"\newcommand{{\TieLineMAE}}{{${sci(exp3['tie_line_mae'])}$}}",
        rf"\newcommand{{\BetaMAE}}{{${sci(exp3['beta_mae'])}$}}",
        rf"\newcommand{{\TPDWorst}}{{${sci(exp3['tpd_worst'])}$}}",
        rf"\newcommand{{\SpearmanRho}}{{{exp4['spearman_rho']:.3f}}}",
        rf"\newcommand{{\OODCapture}}{{{100 * exp4['ood_capture_top_half']:.1f}\%}}",
        rf"\newcommand{{\OODErrorRatio}}{{{exp4['ood_error_ratio']:.2f}}}",
        rf"\newcommand{{\SelectiveFullMAE}}{{${sci(exp4['mae_full_coverage'])}$}}",
        rf"\newcommand{{\SelectiveFortyMAE}}{{${sci(exp4['mae_40pct_coverage'])}$}}",
        rf"\newcommand{{\PermutationResidual}}{{${sci(structural['permutation_residual'])}$}}",
        rf"\newcommand{{\ReferenceRecovery}}{{{100 * exp5['reference_recovery']:.2f}\%}}",
        rf"\newcommand{{\ModelRecovery}}{{{100 * exp5['model_recovery']:.2f}\%}}",
        rf"\newcommand{{\RecoveryError}}{{{100 * exp5['recovery_abs_error']:.3f} percentage points}}",
        rf"\newcommand{{\MassBalanceError}}{{${sci(exp5['max_balance_error'])}$}}",
        rf"\newcommand{{\RuntimeRatio}}{{{exp5['runtime_ratio']:.1f}}}",
    ]
    (SECTIONS / "legacy_results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    started = perf_counter()
    plot_framework()
    binary_params, binary_spec, binary_model, direct, exp1, test_payload = fit_binary_models()
    plot_binary_fit(binary_params, binary_spec, binary_model, test_payload)
    gd = gibbs_duhem_diagnostics(binary_params, binary_spec, binary_model, direct)
    exp2 = binary_binodal_experiment(binary_params, binary_spec, binary_model)
    ternary_params, ternary_spec, ternary_model, ternary_fit = fit_ternary_model()
    exp3 = ternary_flash_experiment(ternary_params, ternary_spec, ternary_model)
    exp4 = uncertainty_experiment(binary_params)
    structural = {"permutation_residual": permutation_equivariance_test(ternary_spec, ternary_model)}
    exp5 = cascade_experiment(ternary_params, ternary_spec, ternary_model)

    metrics = {
        "epistemic_label": "E7 - preliminary numerical results on synthetic NRTL systems",
        "exp1": exp1,
        "gibbs_duhem": gd,
        "exp2": exp2,
        "ternary_fit": ternary_fit,
        "exp3": exp3,
        "exp4": exp4,
        "structural": structural,
        "exp5": exp5,
        "runtime_seconds_total": float(perf_counter() - started),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "autograd": getattr(autograd, "__version__", "not-exposed"),
        },
    }
    with (RESULTS / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_macros(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
