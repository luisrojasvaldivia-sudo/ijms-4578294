#!/usr/bin/env python3
"""Execute the reviewer-driven robustness campaign for the IJMS revision.

The original five synthetic experiments remain reproducible through
``experiments/run_all.py``.  This companion campaign targets the inferential
weaknesses identified by three independent reviews: classical and matched-
capacity baselines, training-seed replication, grid refinement, near-boundary
phase classification, uncertainty controls, thermal derivatives, limiting
activity coefficients, stability saturation, and controlled timing.

No experimental chemical datum is invented.  All outputs produced here remain
E7 synthetic evidence and are archived separately in
``results/revision_metrics.json``.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import warnings
from pathlib import Path
from time import perf_counter

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
from autograd import grad as autograd_grad
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import beta as beta_distribution
from scipy.stats import spearmanr, wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_all import (  # noqa: E402
    COLORS,
    binary_parameters,
    composition_samples_binary,
    composition_samples_ternary,
    make_g_functions,
    nrtl_predict,
    order_phases,
    save_figure,
    ternary_parameters,
)
from src.thermo import (  # noqa: E402
    PairPotentialSpec,
    canonical_embeddings,
    fit_pair_potential,
    flash_two_phase,
    ideal_mixing_energy,
    lower_convex_hull_binodal,
    minimum_tpd_diagnostics,
    nrtl_g_grad_lngamma,
    pair_potential_g_grad,
    pair_potential_predict,
    pair_potential_temperature_derivative,
)


FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "synthetic"
SECTIONS = ROOT / "sections"
for directory in (FIGURES, RESULTS, DATA, SECTIONS):
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 9.8,
        "axes.labelsize": 9.2,
        "legend.fontsize": 7.5,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def percentile_interval(values, probability=0.95):
    values = np.asarray(values, dtype=float)
    alpha = 0.5 * (1.0 - probability)
    return [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))]


def median_iqr(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def bootstrap_spearman_interval(score, error, seed=0, repetitions=2000):
    score = np.asarray(score, dtype=float)
    error = np.asarray(error, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        selection = rng.integers(0, len(score), size=len(score))
        estimates[index] = spearmanr(score[selection], error[selection]).statistic
    return percentile_interval(estimates), estimates


def clopper_pearson(successes, trials, probability=0.95):
    alpha = 1.0 - probability
    lower = 0.0 if successes == 0 else beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else beta_distribution.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    return [float(lower), float(upper)]


# ---------------------------------------------------------------------------
# Classical Redlich--Kister and refitted-NRTL baselines
# ---------------------------------------------------------------------------


def redlich_kister_design(x, temperature, composition_terms=5, temperature_terms=3, center=370.0, scale=80.0):
    """Linear design matrix mapping Redlich--Kister coefficients to ln(gamma)."""

    x = np.asarray(x, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    x1 = x[:, 0]
    x2 = x[:, 1]
    u = x1 - x2
    t = (temperature - center) / scale
    columns_component_1 = []
    columns_component_2 = []
    columns_g = []
    for k in range(composition_terms):
        for m in range(temperature_terms):
            temperature_basis = t**m
            composition_basis = u**k
            g_basis = x1 * x2 * composition_basis * temperature_basis
            if k == 0:
                derivative_u = np.zeros_like(u)
            else:
                derivative_u = k * u ** (k - 1)
            dg_dx1 = (
                (x2 - x1) * composition_basis
                + 2.0 * x1 * x2 * derivative_u
            ) * temperature_basis
            columns_g.append(g_basis)
            columns_component_1.append(g_basis + x2 * dg_dx1)
            columns_component_2.append(g_basis - x1 * dg_dx1)
    design_1 = np.column_stack(columns_component_1)
    design_2 = np.column_stack(columns_component_2)
    design = np.empty((2 * len(x), design_1.shape[1]))
    design[0::2] = design_1
    design[1::2] = design_2
    return design, np.column_stack(columns_g)


def fit_redlich_kister(x, temperature, target, ridge=1.0e-12):
    design, _ = redlich_kister_design(x, temperature)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ np.asarray(target).reshape(-1))


def predict_redlich_kister(coefficients, x, temperature):
    design, g_design = redlich_kister_design(x, temperature)
    prediction = (design @ coefficients).reshape(len(x), 2)
    g_excess = g_design @ coefficients
    return g_excess, prediction


def fit_symmetric_nrtl(x, temperature, target, free_alpha=False):
    target = np.asarray(target, dtype=float)

    def decode(parameters):
        b12, b21 = parameters[:2]
        alpha_value = float(parameters[2]) if free_alpha else 0.20
        tau_constant = np.zeros((2, 2))
        tau_inverse = np.array([[0.0, b12], [b21, 0.0]])
        alpha = np.array([[0.0, alpha_value], [alpha_value, 0.0]])
        return tau_constant, tau_inverse, alpha

    def residual(parameters):
        return (nrtl_g_grad_lngamma(x, temperature, *decode(parameters))[2] - target).reshape(-1)

    initial = np.array([780.0, 900.0, 0.20]) if free_alpha else np.array([780.0, 900.0])
    lower = np.array([100.0, 100.0, 0.05]) if free_alpha else np.array([100.0, 100.0])
    upper = np.array([1800.0, 1800.0, 0.50]) if free_alpha else np.array([1800.0, 1800.0])
    result = least_squares(residual, initial, bounds=(lower, upper), xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11, max_nfev=2500)
    return result.x, decode(result.x), result


def uniquac_predict(x, temperature, parameters):
    """Binary UNIQUAC activity coefficients with fitted size and energy terms."""

    x = np.asarray(x, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    r = np.array([1.0, parameters[0]])
    q = np.array([1.0, parameters[1]])
    interaction_energy = np.array([[0.0, parameters[2]], [parameters[3], 0.0]])
    tau = np.exp(-interaction_energy[None, :, :] / temperature[:, None, None])
    phi = x * r[None, :] / np.sum(x * r[None, :], axis=1, keepdims=True)
    theta = x * q[None, :] / np.sum(x * q[None, :], axis=1, keepdims=True)
    z = 10.0
    ell = 0.5 * z * (r - q) - (r - 1.0)
    combinatorial = (
        np.log(np.clip(phi / x, 1.0e-300, None))
        + 0.5 * z * q[None, :] * np.log(np.clip(theta / phi, 1.0e-300, None))
        + ell[None, :]
        - (phi / x) * np.sum(x * ell[None, :], axis=1, keepdims=True)
    )
    # denominator[n,j] = sum_k theta_k tau_kj
    denominator = np.einsum("nk,nkj->nj", theta, tau)
    first = np.einsum("nj,nji->ni", theta, tau)
    second = np.zeros_like(x)
    for i in range(2):
        second[:, i] = np.sum(theta * tau[:, i, :] / denominator, axis=1)
    residual = q[None, :] * (1.0 - np.log(np.clip(first, 1.0e-300, None)) - second)
    return combinatorial + residual


def fit_uniquac(x, temperature, target):
    def residual(parameters):
        return (uniquac_predict(x, temperature, parameters) - target).reshape(-1)

    result = least_squares(
        residual,
        x0=np.array([1.2, 1.1, 300.0, 300.0]),
        bounds=(np.array([0.2, 0.2, -2500.0, -2500.0]), np.array([8.0, 8.0, 2500.0, 2500.0])),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=5000,
    )
    return result.x, result


def uniquac_supplemental_campaign():
    binary = binary_parameters()
    x_train, t_train = composition_samples_binary(np.random.default_rng(42), 420, 290.0, 450.0)
    x_test, t_test = composition_samples_binary(np.random.default_rng(99), 2400, 290.0, 450.0)
    clean = nrtl_predict(x_train, t_train, binary)[2]
    target = nrtl_predict(x_test, t_test, binary)[2]
    rows = []
    for seed in range(10):
        noisy = clean + np.random.default_rng(1000 + seed).normal(scale=0.01, size=clean.shape)
        parameters, fit = fit_uniquac(x_train, t_train, noisy)
        prediction = uniquac_predict(x_test, t_test, parameters)
        rows.append(
            {
                "seed": seed,
                "test_mae": float(np.mean(np.abs(prediction - target))),
                "parameters_r2_q2_u12_u21_K": parameters.tolist(),
                "cost": float(fit.cost),
                "function_evaluations": int(fit.nfev),
                "converged": bool(fit.success),
            }
        )
    summary = median_iqr([row["test_mae"] for row in rows])
    payload = {"model": "binary UNIQUAC with r1=q1=1 and fitted r2,q2,u12,u21", "replicates": rows, "summary": summary}
    with (RESULTS / "uniquac_seed_replicates.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    (SECTIONS / "uniquac_macro.tex").write_text(
        "% Auto-generated supplemental classical baseline.\n"
        + rf"\newcommand{{\UNIQUACMedianMAE}}{{${_sci(summary['median'])}$}}" + "\n",
        encoding="utf-8",
    )
    return payload


# ---------------------------------------------------------------------------
# Matched-capacity direct and Gibbs--Duhem-informed neural baselines
# ---------------------------------------------------------------------------


def initialize_matched_mlp(hidden_units, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=np.sqrt(2.0 / (hidden_units + 2)), size=(hidden_units, 2))
    b = rng.normal(scale=0.03, size=hidden_units)
    v = rng.normal(scale=0.08, size=(2, hidden_units))
    c = np.zeros(2)
    return np.concatenate([w.ravel(), b, v.ravel(), c])


def unpack_matched_mlp(parameters, hidden_units):
    cursor = 0
    w = parameters[cursor : cursor + 2 * hidden_units].reshape(hidden_units, 2)
    cursor += 2 * hidden_units
    b = parameters[cursor : cursor + hidden_units]
    cursor += hidden_units
    v = parameters[cursor : cursor + 2 * hidden_units].reshape(2, hidden_units)
    cursor += 2 * hidden_units
    c = parameters[cursor : cursor + 2]
    return w, b, v, c


def matched_mlp_core(parameters, x, temperature, hidden_units, center=370.0, scale=80.0):
    x1 = x[:, 0]
    features = anp.column_stack([2.0 * x1 - 1.0, (temperature - center) / scale])
    w, b, v, c = unpack_matched_mlp(parameters, hidden_units)
    hidden = anp.tanh(anp.dot(features, w.T) + b)
    prediction = anp.dot(hidden, v.T) + c
    d_hidden_dx1 = 2.0 * (1.0 - hidden**2) * w[:, 0]
    derivative = anp.dot(d_hidden_dx1, v.T)
    gd_residual = x1 * derivative[:, 0] + (1.0 - x1) * derivative[:, 1]
    return prediction, derivative, gd_residual


def fit_matched_mlp(x, temperature, target, *, hidden_units=16, seed=0, gd_weight=0.0, l2_weight=1.0e-7, maxiter=750):
    x = np.asarray(x, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    target = np.asarray(target, dtype=float)
    initial = initialize_matched_mlp(hidden_units, seed)

    def objective(parameters):
        prediction, _, gd = matched_mlp_core(parameters, anp.asarray(x), anp.asarray(temperature), hidden_units)
        return anp.mean((prediction - target) ** 2) + gd_weight * anp.mean(gd**2) + l2_weight * anp.mean(parameters**2)

    gradient = autograd_grad(objective)
    started = perf_counter()
    result = minimize(
        lambda p: float(objective(p)),
        initial,
        jac=lambda p: np.asarray(gradient(p), dtype=float),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1.0e-10, "gtol": 1.0e-7, "maxls": 40},
    )
    return {
        "parameters": np.asarray(result.x),
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "seconds": float(perf_counter() - started),
        "objective": float(result.fun),
        "message": str(result.message),
        "hidden_units": hidden_units,
        "gd_weight": gd_weight,
        "parameter_count": int(len(result.x)),
    }


def predict_matched_mlp(fit, x, temperature):
    prediction, derivative, gd = matched_mlp_core(
        fit["parameters"], anp.asarray(x), anp.asarray(temperature), fit["hidden_units"]
    )
    return np.asarray(prediction), np.asarray(derivative), np.asarray(gd)


def fit_fixed_gp(x, temperature, target):
    features = np.column_stack([2.0 * x[:, 0] - 1.0, (temperature - 370.0) / 80.0])
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF([0.35, 0.75], length_scale_bounds="fixed") + WhiteKernel(1.0e-4, noise_level_bounds="fixed")
    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, alpha=1.0e-8, normalize_y=True, optimizer=None),
    )
    model.fit(features, target)
    return model


def predict_fixed_gp(model, x, temperature, return_std=False):
    features = np.column_stack([2.0 * x[:, 0] - 1.0, (temperature - 370.0) / 80.0])
    if return_std:
        return model.predict(features, return_std=True)
    return model.predict(features)


def finite_difference_gd(predictor, grid_size, temperature=300.0):
    x1 = np.linspace(0.003, 0.997, grid_size)
    x = np.column_stack([x1, 1.0 - x1])
    t = np.full(grid_size, temperature)
    values = np.asarray(predictor(x, t))
    derivative = np.gradient(values, x1, axis=0, edge_order=2)
    residual = np.abs(np.sum(x * derivative, axis=1))
    trim = max(4, int(round(0.01 * grid_size)))
    return float(np.mean(residual[trim:-trim])), float(x1[1] - x1[0])


def binary_baseline_campaign():
    binary = binary_parameters()
    x_train, t_train = composition_samples_binary(np.random.default_rng(42), 420, 290.0, 450.0)
    x_validation, t_validation = composition_samples_binary(np.random.default_rng(43), 240, 290.0, 450.0)
    x_test, t_test = composition_samples_binary(np.random.default_rng(99), 2400, 290.0, 450.0)
    _, _, y_train_clean = nrtl_predict(x_train, t_train, binary)
    _, _, y_validation = nrtl_predict(x_validation, t_validation, binary)
    g_test, _, y_test = nrtl_predict(x_test, t_test, binary)

    # Hyperparameters are selected once on a validation set that is disjoint
    # from both training and test states, then frozen across seed replications.
    tuning_rng = np.random.default_rng(700)
    tuning_target = y_train_clean + tuning_rng.normal(scale=0.01, size=y_train_clean.shape)
    l2_candidates = [0.0, 1.0e-9, 3.0e-8, 1.0e-7]
    potential_tuning = []
    for l2 in l2_candidates:
        spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=16, temperature_center=370.0, temperature_scale=80.0)
        fit = fit_pair_potential(spec, x_train, t_train, tuning_target, seed=701, maxiter=650, l2_weight=l2)
        prediction = pair_potential_predict(fit.parameters, spec, x_validation, t_validation)[2]
        potential_tuning.append({"l2": l2, "validation_mae": float(np.mean(np.abs(prediction - y_validation)))})
    selected_l2 = min(potential_tuning, key=lambda row: row["validation_mae"])["l2"]

    soft_candidates = [1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    soft_tuning = []
    for gd_weight in soft_candidates:
        fit = fit_matched_mlp(x_train, t_train, tuning_target, hidden_units=16, seed=702, gd_weight=gd_weight)
        prediction, _, gd = predict_matched_mlp(fit, x_validation, t_validation)
        soft_tuning.append(
            {
                "gd_weight": gd_weight,
                "validation_mae": float(np.mean(np.abs(prediction - y_validation))),
                "validation_gd": float(np.mean(np.abs(gd))),
            }
        )
    eligible = [row for row in soft_tuning if row["validation_gd"] <= 5.0e-3]
    selected_soft_weight = min(eligible or soft_tuning, key=lambda row: row["validation_mae"])["gd_weight"]

    records = []
    first_models = None
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        noisy = y_train_clean + rng.normal(scale=0.01, size=y_train_clean.shape)
        spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=16, temperature_center=370.0, temperature_scale=80.0)
        potential = fit_pair_potential(spec, x_train, t_train, noisy, seed=2000 + seed, maxiter=650, l2_weight=selected_l2)
        g_potential, _, y_potential = pair_potential_predict(potential.parameters, spec, x_test, t_test)
        direct = fit_matched_mlp(x_train, t_train, noisy, hidden_units=16, seed=3000 + seed, gd_weight=0.0)
        y_direct, _, gd_direct_test = predict_matched_mlp(direct, x_test, t_test)
        soft = fit_matched_mlp(x_train, t_train, noisy, hidden_units=16, seed=4000 + seed, gd_weight=selected_soft_weight)
        y_soft, _, gd_soft_test = predict_matched_mlp(soft, x_test, t_test)
        rk_coefficients = fit_redlich_kister(x_train, t_train, noisy)
        g_rk, y_rk = predict_redlich_kister(rk_coefficients, x_test, t_test)
        nrtl_fixed_vector, nrtl_fixed, nrtl_fixed_fit = fit_symmetric_nrtl(x_train, t_train, noisy, free_alpha=False)
        y_nrtl_fixed = nrtl_predict(x_test, t_test, nrtl_fixed)[2]
        nrtl_free_vector, nrtl_free, nrtl_free_fit = fit_symmetric_nrtl(x_train, t_train, noisy, free_alpha=True)
        y_nrtl_free = nrtl_predict(x_test, t_test, nrtl_free)[2]
        gp = fit_fixed_gp(x_train, t_train, noisy)
        y_gp = predict_fixed_gp(gp, x_test, t_test)

        predictions = {
            "potential": y_potential,
            "direct_matched": y_direct,
            "soft_gd": y_soft,
            "redlich_kister": y_rk,
            "nrtl_fixed_alpha": y_nrtl_fixed,
            "nrtl_free_alpha": y_nrtl_free,
            "gaussian_process": y_gp,
        }
        record = {"seed": seed}
        for name, prediction in predictions.items():
            record[f"{name}_mae"] = float(np.mean(np.abs(prediction - y_test)))
        record.update(
            {
                "potential_g_mae": float(np.mean(np.abs(g_potential - g_test))),
                "rk_g_mae": float(np.mean(np.abs(g_rk - g_test))),
                "direct_gd_path_mae": float(np.mean(np.abs(gd_direct_test))),
                "soft_gd_path_mae": float(np.mean(np.abs(gd_soft_test))),
                "potential_converged": potential.converged,
                "direct_converged": direct["converged"],
                "soft_converged": soft["converged"],
                "nrtl_fixed_parameters": nrtl_fixed_vector.tolist(),
                "nrtl_free_parameters": nrtl_free_vector.tolist(),
                "nrtl_fixed_cost": float(nrtl_fixed_fit.cost),
                "nrtl_free_cost": float(nrtl_free_fit.cost),
            }
        )
        records.append(record)
        if seed == 0:
            first_models = {
                "spec": spec,
                "potential": potential.parameters,
                "direct": direct,
                "soft": soft,
                "rk": rk_coefficients,
                "nrtl_fixed": nrtl_fixed,
                "nrtl_free": nrtl_free,
                "gp": gp,
                "x_train": x_train,
                "t_train": t_train,
                "y_train_clean": y_train_clean,
                "x_validation": x_validation,
                "t_validation": t_validation,
                "x_test": x_test,
                "t_test": t_test,
                "y_test": y_test,
                "g_test": g_test,
            }

    methods = [
        "potential",
        "direct_matched",
        "soft_gd",
        "redlich_kister",
        "nrtl_fixed_alpha",
        "nrtl_free_alpha",
        "gaussian_process",
    ]
    summary = {method: median_iqr([row[f"{method}_mae"] for row in records]) for method in methods}
    paired_tests = {}
    potential_values = np.array([row["potential_mae"] for row in records])
    for method in methods[1:]:
        comparator = np.array([row[f"{method}_mae"] for row in records])
        result = wilcoxon(potential_values, comparator, alternative="less", zero_method="wilcox", method="auto")
        paired_tests[method] = {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}

    parameter_counts = {
        "potential_stored": first_models["spec"].parameter_count,
        "potential_identifiable_binary": 81,
        "direct_matched": first_models["direct"]["parameter_count"],
        "soft_gd": first_models["soft"]["parameter_count"],
        "redlich_kister": 15,
        "nrtl_fixed_alpha": 2,
        "nrtl_free_alpha": 3,
        "gaussian_process_hyperparameters_fixed": 4,
    }

    # Seed-level comparison replaces the invalid state-level significance test.
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    values = [[row[f"{method}_mae"] for row in records] for method in methods]
    labels = ["Hard potential", "Direct\nmatched", "Soft GD", "Redlich-\nKister", "NRTL\nfixed alpha", "NRTL\nfree alpha", "Gaussian\nprocess"]
    parts = ax.violinplot(values, showmedians=True, showextrema=False)
    for index, body in enumerate(parts["bodies"]):
        body.set_facecolor([COLORS["potential"], COLORS["direct"], COLORS["uncertainty"], "#e9c46a", "#457b9d", "#1d3557", "#6c757d"][index])
        body.set_alpha(0.72)
    parts["cmedians"].set_color("#202020")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_yscale("log")
    ax.set_ylabel(r"Held-out MAE of $\ln\gamma_i$")
    ax.set_title("Ten independent training-noise and initialization seeds")
    ax.grid(axis="y", which="both", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, "fig08_baseline_replicates")

    with (RESULTS / "binary_seed_replicates.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, allow_nan=False)
    return {
        "selected_l2": selected_l2,
        "selected_soft_gd_weight": selected_soft_weight,
        "potential_tuning": potential_tuning,
        "soft_tuning": soft_tuning,
        "replicates": records,
        "summary": summary,
        "paired_tests": paired_tests,
        "parameter_counts": parameter_counts,
    }, first_models


def binary_residual_figure(models):
    x_axis = np.linspace(0.004, 0.996, 150)
    t_axis = np.linspace(290.0, 450.0, 100)
    xx, tt = np.meshgrid(x_axis, t_axis)
    x = np.column_stack([xx.ravel(), 1.0 - xx.ravel()])
    temperature = tt.ravel()
    target = nrtl_predict(x, temperature, binary_parameters())[2]
    potential = pair_potential_predict(models["potential"], models["spec"], x, temperature)[2]
    direct = predict_matched_mlp(models["direct"], x, temperature)[0]
    rk = predict_redlich_kister(models["rk"], x, temperature)[1]
    errors = [
        np.mean(np.abs(potential - target), axis=1).reshape(tt.shape),
        np.mean(np.abs(direct - target), axis=1).reshape(tt.shape),
        np.mean(np.abs(rk - target), axis=1).reshape(tt.shape),
    ]
    log_errors = [np.log10(error + 1.0e-7) for error in errors]
    vmin = min(np.min(value) for value in log_errors)
    vmax = max(np.max(value) for value in log_errors)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharex=True, sharey=True)
    titles = ["(a) Hard potential", "(b) Direct matched", "(c) Redlich-Kister"]
    image = None
    for ax, value, title in zip(axes, log_errors, titles):
        image = ax.pcolormesh(x_axis, t_axis, value, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"Mole fraction $x_1$")
    axes[0].set_ylabel("Temperature (K)")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label(r"$\log_{10}$ statewise MAE")
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.18, top=0.86, wspace=0.10)
    save_figure(fig, "fig02_binary_fit")


def baseline_property_campaign(models):
    """Audit dilute limits, GD, and a common operational hull for every baseline."""

    binary = binary_parameters()
    noisy = models["y_train_clean"] + np.random.default_rng(1000).normal(scale=0.01, size=models["y_train_clean"].shape)
    uniquac_parameters, _ = fit_uniquac(models["x_train"], models["t_train"], noisy)
    ln_predictors = {
        "hard_potential": lambda x, t: pair_potential_predict(models["potential"], models["spec"], x, t)[2],
        "direct_matched": lambda x, t: predict_matched_mlp(models["direct"], x, t)[0],
        "soft_gd": lambda x, t: predict_matched_mlp(models["soft"], x, t)[0],
        "redlich_kister": lambda x, t: predict_redlich_kister(models["rk"], x, t)[1],
        "nrtl_fixed_alpha": lambda x, t: nrtl_predict(x, t, models["nrtl_fixed"])[2],
        "nrtl_free_alpha": lambda x, t: nrtl_predict(x, t, models["nrtl_free"])[2],
        "gaussian_process": lambda x, t: predict_fixed_gp(models["gp"], x, t),
        "uniquac": lambda x, t: uniquac_predict(x, t, uniquac_parameters),
    }
    g_predictors = {
        "hard_potential": lambda x, t: pair_potential_predict(models["potential"], models["spec"], x, t)[0],
        "redlich_kister": lambda x, t: predict_redlich_kister(models["rk"], x, t)[0],
        "nrtl_fixed_alpha": lambda x, t: nrtl_predict(x, t, models["nrtl_fixed"])[0],
        "nrtl_free_alpha": lambda x, t: nrtl_predict(x, t, models["nrtl_free"])[0],
    }
    # For direct-output, GP, soft-GD, and UNIQUAC controls, x dot ln(gamma)
    # is used as an explicitly labeled operational free-energy reconstruction.
    for name in ("direct_matched", "soft_gd", "gaussian_process", "uniquac"):
        g_predictors[name] = lambda x, t, method=name: np.sum(x * ln_predictors[method](x, t), axis=1)

    temperatures = np.array([300.0, 350.0, 400.0, 440.0])
    epsilon = 1.0e-8
    dilute_x = np.array([[epsilon, 1.0 - epsilon], [1.0 - epsilon, epsilon]] * len(temperatures))
    dilute_t = np.repeat(temperatures, 2)
    dilute_reference = nrtl_predict(dilute_x, dilute_t, binary)[2]
    dilute_components = np.array([0, 1] * len(temperatures))
    metrics = {}
    for name, predictor in ln_predictors.items():
        dilute_prediction = predictor(dilute_x, dilute_t)
        selected_prediction = dilute_prediction[np.arange(len(dilute_x)), dilute_components]
        selected_reference = dilute_reference[np.arange(len(dilute_x)), dilute_components]
        gd, _ = finite_difference_gd(predictor, 4801, temperature=300.0)
        metrics[name] = {
            "infinite_dilution_mae": float(np.mean(np.abs(selected_prediction - selected_reference))),
            "infinite_dilution_max": float(np.max(np.abs(selected_prediction - selected_reference))),
            "gibbs_duhem_residual_n4801": gd,
            "binodal_g_definition": "native scalar potential" if name in {"hard_potential", "redlich_kister", "nrtl_fixed_alpha", "nrtl_free_alpha"} else "operational sum_i x_i ln(gamma_i)",
        }

    x1 = np.linspace(0.001, 0.999, 10400)
    x_grid = np.column_stack([x1, 1.0 - x1])
    for name, predictor in g_predictors.items():
        errors = []
        for temperature in np.linspace(292.0, 448.0, 20):
            t = np.full(len(x_grid), temperature)
            reference_endpoint = lower_convex_hull_binodal(x1, ideal_mixing_energy(x_grid) + nrtl_predict(x_grid, t, binary)[0])
            method_endpoint = lower_convex_hull_binodal(x1, ideal_mixing_energy(x_grid) + predictor(x_grid, t))
            if reference_endpoint is not None and method_endpoint is not None:
                errors.extend(np.abs(np.asarray(reference_endpoint) - np.asarray(method_endpoint)).tolist())
        metrics[name]["binodal_matched_isotherms"] = len(errors) // 2
        metrics[name]["binodal_endpoint_mae_10400"] = float(np.mean(errors)) if errors else None
        metrics[name]["binodal_endpoint_max_10400"] = float(np.max(errors)) if errors else None
    payload = {
        "seed": 0,
        "isotherms": 20,
        "grid_size": 10400,
        "note": "Operational hulls for direct-output models use gE=sum_i x_i ln(gamma_i); this does not restore integrability.",
        "methods": metrics,
    }
    with (RESULTS / "baseline_property_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    labels = {
        "hard_potential": "Hard potential",
        "direct_matched": "Direct matched",
        "soft_gd": "Soft GD",
        "redlich_kister": "Redlich--Kister",
        "nrtl_fixed_alpha": "NRTL, fixed $\\alpha$",
        "nrtl_free_alpha": "NRTL, free $\\alpha$",
        "gaussian_process": "Gaussian process",
        "uniquac": "UNIQUAC",
    }
    table_lines = [
        "% Auto-generated by baseline_property_campaign; do not edit.",
        r"\begin{table}[H]",
        r"\caption{Seed-0 thermodynamic-property audit for every binary baseline. Direct-output models use the explicitly labeled operational reconstruction $g^E=\sum_i x_i\ln\gamma_i$ for the hull; this does not restore integrability.\label{tab:baseline_properties}}",
        r"\begin{adjustwidth}{-\extralength}{0cm}",
        r"\small",
        r"\begin{tabularx}{\fulllength}{LCCC}",
        r"\toprule",
        r"\textbf{Method} & \textbf{$\ln\gamma^\infty$ MAE} & \textbf{GD residual ($N=4801$)} & \textbf{Binodal endpoint MAE}\\",
        r"\midrule",
    ]
    for name, row in metrics.items():
        table_lines.append(
            f"{labels[name]} & ${_sci(row['infinite_dilution_mae'])}$ & ${_sci(row['gibbs_duhem_residual_n4801'])}$ & ${_sci(row['binodal_endpoint_mae_10400'])}$\\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"])
    (SECTIONS / "baseline_property_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    return payload


def gibbs_duhem_refinement(models):
    grid_sizes = [301, 601, 1201, 2401, 4801]
    predictors = {
        "nrtl": lambda x, t: nrtl_predict(x, t, binary_parameters())[2],
        "potential": lambda x, t: pair_potential_predict(models["potential"], models["spec"], x, t)[2],
        "redlich_kister": lambda x, t: predict_redlich_kister(models["rk"], x, t)[1],
        "soft_gd": lambda x, t: predict_matched_mlp(models["soft"], x, t)[0],
        "direct": lambda x, t: predict_matched_mlp(models["direct"], x, t)[0],
    }
    rows = []
    for n in grid_sizes:
        row = {"grid_size": n}
        for name, predictor in predictors.items():
            residual, step = finite_difference_gd(predictor, n, temperature=300.0)
            row[name] = residual
            row["step"] = step
        rows.append(row)
    slopes = {}
    h = np.array([row["step"] for row in rows])
    for name in ("nrtl", "potential", "redlich_kister"):
        values = np.array([row[name] for row in rows])
        slopes[name] = float(np.polyfit(np.log(h), np.log(values), 1)[0])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    styles = {
        "nrtl": (COLORS["reference"], "o", "Analytic NRTL"),
        "potential": (COLORS["potential"], "s", "Hard potential"),
        "redlich_kister": ("#e9c46a", "^", "Redlich-Kister"),
        "soft_gd": (COLORS["uncertainty"], "D", "Soft GD network"),
        "direct": (COLORS["direct"], "v", "Direct matched network"),
    }
    for name, (color, marker, label) in styles.items():
        ax.loglog(h, [row[name] for row in rows], marker=marker, color=color, label=label)
    ax.invert_xaxis()
    ax.set_xlabel("Composition-grid spacing h")
    ax.set_ylabel("Mean finite-difference Gibbs-Duhem residual")
    ax.set_title("Numerical certificate under grid refinement")
    ax.grid(which="both", alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig03_gibbs_duhem")
    return {"rows": rows, "second_order_slopes": slopes}


def binodal_refinement(models):
    temperatures = np.linspace(292.0, 448.0, 40)
    grid_sizes = [2600, 10400, 41600]
    convergence = []
    final_reference = None
    final_prediction = None
    for grid_size in grid_sizes:
        x1_grid = np.linspace(0.001, 0.999, grid_size)
        x_grid = np.column_stack([x1_grid, 1.0 - x1_grid])
        reference = []
        prediction = []
        for temperature in temperatures:
            t = np.full(grid_size, temperature)
            g_ref = nrtl_predict(x_grid, t, binary_parameters())[0]
            g_pred = pair_potential_predict(models["potential"], models["spec"], x_grid, t)[0]
            ref_endpoints = lower_convex_hull_binodal(x1_grid, ideal_mixing_energy(x_grid) + g_ref)
            pred_endpoints = lower_convex_hull_binodal(x1_grid, ideal_mixing_energy(x_grid) + g_pred)
            if ref_endpoints is not None and pred_endpoints is not None:
                reference.append([temperature, *ref_endpoints])
                prediction.append([temperature, *pred_endpoints])
        reference = np.asarray(reference)
        prediction = np.asarray(prediction)
        endpoint_errors = np.abs(reference[:, 1:] - prediction[:, 1:])
        convergence.append(
            {
                "grid_size": grid_size,
                "spacing": float(x1_grid[1] - x1_grid[0]),
                "endpoint_mae": float(np.mean(endpoint_errors)),
                "composition_hausdorff_mean": float(np.mean(np.max(endpoint_errors, axis=1))),
                "composition_hausdorff_max": float(np.max(endpoint_errors)),
            }
        )
        final_reference, final_prediction = reference, prediction

    endpoint_residual = final_prediction[:, 1:] - final_reference[:, 1:]
    # Invert each monotone branch to express the remaining discrepancy in kelvin.
    temperature_errors = []
    for column in (1, 2):
        ref_x = final_reference[:, column]
        pred_x = final_prediction[:, column]
        ref_order = np.argsort(ref_x)
        pred_order = np.argsort(pred_x)
        lower = max(np.min(ref_x), np.min(pred_x))
        upper = min(np.max(ref_x), np.max(pred_x))
        query = np.linspace(lower, upper, 200)
        ref_t = np.interp(query, ref_x[ref_order], final_reference[ref_order, 0])
        pred_t = np.interp(query, pred_x[pred_order], final_prediction[pred_order, 0])
        temperature_errors.extend(np.abs(ref_t - pred_t).tolist())

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75))
    axes[0].plot(final_reference[:, 1], final_reference[:, 0], color=COLORS["reference"], label="NRTL reference")
    axes[0].plot(final_reference[:, 2], final_reference[:, 0], color=COLORS["reference"])
    axes[0].plot(final_prediction[:, 1], final_prediction[:, 0], "--", color=COLORS["potential"], label="Hard potential")
    axes[0].plot(final_prediction[:, 2], final_prediction[:, 0], "--", color=COLORS["potential"])
    axes[0].set_xlabel(r"Mole fraction $x_1$")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("(a) Fine-grid binodal")
    axes[0].legend(frameon=False, loc="lower center")
    axes[1].plot(final_reference[:, 0], endpoint_residual[:, 0], marker="o", ms=3, color="#0072B2", label="Left branch")
    axes[1].plot(final_reference[:, 0], endpoint_residual[:, 1], marker="s", ms=3, color="#D55E00", label="Right branch")
    axes[1].axhline(0.0, color="#333333", lw=0.7)
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel(r"Endpoint residual $\Delta x_1$")
    axes[1].set_title("(b) Branch-resolved error")
    axes[1].legend(frameon=False)
    axes[2].loglog([row["spacing"] for row in convergence], [row["endpoint_mae"] for row in convergence], marker="o", color=COLORS["potential"])
    axes[2].invert_xaxis()
    axes[2].set_xlabel("Grid spacing")
    axes[2].set_ylabel("Endpoint MAE")
    axes[2].set_title("(c) Measurement convergence")
    axes[2].grid(which="both", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, "fig04_binary_binodal")

    np.savetxt(DATA / "binary_binodal_reference_fine.csv", final_reference, delimiter=",", header="temperature_K,x1_left,x1_right", comments="")
    np.savetxt(DATA / "binary_binodal_prediction_fine.csv", final_prediction, delimiter=",", header="temperature_K,x1_left,x1_right", comments="")
    return {
        "grid_convergence": convergence,
        "fine_grid_endpoint_mae": float(np.mean(np.abs(endpoint_residual))),
        "fine_grid_composition_hausdorff_mean": float(np.mean(np.max(np.abs(endpoint_residual), axis=1))),
        "fine_grid_composition_hausdorff_max": float(np.max(np.abs(endpoint_residual))),
        "inverse_temperature_mae_K": float(np.mean(temperature_errors)),
        "isotherms": int(len(final_reference)),
    }


def data_efficiency_noise_and_ablation(models, selected_l2):
    binary = binary_parameters()
    x_pool, t_pool = composition_samples_binary(np.random.default_rng(510), 1680, 290.0, 450.0)
    y_pool = nrtl_predict(x_pool, t_pool, binary)[2]
    x_test, t_test, y_test = models["x_test"], models["t_test"], models["y_test"]
    sizes = [105, 210, 420, 840, 1680]
    efficiency = []
    for size in sizes:
        method_values = {"potential": [], "redlich_kister": [], "nrtl_fixed": []}
        for seed in range(10):
            rng = np.random.default_rng(6000 + 100 * size + seed)
            noisy = y_pool[:size] + rng.normal(scale=0.01, size=y_pool[:size].shape)
            spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=16)
            fit = fit_pair_potential(spec, x_pool[:size], t_pool[:size], noisy, seed=7000 + seed, maxiter=550, l2_weight=selected_l2)
            pred = pair_potential_predict(fit.parameters, spec, x_test, t_test)[2]
            rk = fit_redlich_kister(x_pool[:size], t_pool[:size], noisy)
            rk_pred = predict_redlich_kister(rk, x_test, t_test)[1]
            _, nrtl_fit, _ = fit_symmetric_nrtl(x_pool[:size], t_pool[:size], noisy, free_alpha=False)
            nrtl_pred = nrtl_predict(x_test, t_test, nrtl_fit)[2]
            method_values["potential"].append(float(np.mean(np.abs(pred - y_test))))
            method_values["redlich_kister"].append(float(np.mean(np.abs(rk_pred - y_test))))
            method_values["nrtl_fixed"].append(float(np.mean(np.abs(nrtl_pred - y_test))))
        efficiency.append({"train_size": size, **{name: median_iqr(values) for name, values in method_values.items()}})

    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    noise_results = []
    x_train, t_train, clean = models["x_train"], models["t_train"], models["y_train_clean"]
    for sigma in noise_levels:
        ratios = []
        potential_values = []
        direct_values = []
        for seed in range(10):
            rng = np.random.default_rng(8000 + int(1000 * sigma) + seed)
            noisy = clean + rng.normal(scale=sigma, size=clean.shape)
            spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=16)
            fit = fit_pair_potential(spec, x_train, t_train, noisy, seed=8100 + seed, maxiter=550, l2_weight=selected_l2)
            pred = pair_potential_predict(fit.parameters, spec, x_test, t_test)[2]
            direct = fit_matched_mlp(x_train, t_train, noisy, hidden_units=16, seed=8200 + seed, gd_weight=0.0, maxiter=650)
            direct_pred = predict_matched_mlp(direct, x_test, t_test)[0]
            p_mae = float(np.mean(np.abs(pred - y_test)))
            d_mae = float(np.mean(np.abs(direct_pred - y_test)))
            potential_values.append(p_mae)
            direct_values.append(d_mae)
            ratios.append(d_mae / p_mae)
        noise_results.append({"sigma": sigma, "potential": median_iqr(potential_values), "direct": median_iqr(direct_values), "advantage_ratio": median_iqr(ratios)})

    heteroscedastic_ratios = []
    for seed in range(10):
        rng = np.random.default_rng(9000 + seed)
        scale = 0.003 + 0.006 * np.abs(clean)
        noisy = clean + rng.normal(scale=scale, size=clean.shape)
        spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=16)
        fit = fit_pair_potential(spec, x_train, t_train, noisy, seed=9100 + seed, maxiter=550, l2_weight=selected_l2)
        pred = pair_potential_predict(fit.parameters, spec, x_test, t_test)[2]
        direct = fit_matched_mlp(x_train, t_train, noisy, hidden_units=16, seed=9200 + seed, gd_weight=0.0, maxiter=650)
        direct_pred = predict_matched_mlp(direct, x_test, t_test)[0]
        heteroscedastic_ratios.append(float(np.mean(np.abs(direct_pred - y_test)) / np.mean(np.abs(pred - y_test))))

    # Capacity/lambda ablation on the untouched validation set.
    hidden_values = [8, 16, 24, 32]
    l2_values = [0.0, 1.0e-9, 3.0e-8, 1.0e-7]
    ablation = []
    rng = np.random.default_rng(9300)
    noisy = clean + rng.normal(scale=0.01, size=clean.shape)
    for hidden in hidden_values:
        for l2 in l2_values:
            spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=hidden)
            fit = fit_pair_potential(spec, x_train, t_train, noisy, seed=9400 + hidden, maxiter=650, l2_weight=l2)
            pred = pair_potential_predict(fit.parameters, spec, models["x_validation"], models["t_validation"])[2]
            ablation.append({"hidden_units": hidden, "l2": l2, "validation_mae": float(np.mean(np.abs(pred - nrtl_predict(models["x_validation"], models["t_validation"], binary)[2])))})

    fig, axes = plt.subplots(1, 3, figsize=(7.7, 2.8))
    for method, color, label in (("potential", COLORS["potential"], "Hard potential"), ("redlich_kister", "#e9c46a", "Redlich-Kister"), ("nrtl_fixed", COLORS["reference"], "Refitted NRTL")):
        med = [row[method]["median"] for row in efficiency]
        low = [row[method]["q1"] for row in efficiency]
        high = [row[method]["q3"] for row in efficiency]
        axes[0].plot(sizes, med, marker="o", color=color, label=label)
        axes[0].fill_between(sizes, low, high, color=color, alpha=0.14)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training states")
    axes[0].set_ylabel("Test MAE")
    axes[0].set_title("(a) Data efficiency")
    axes[0].legend(frameon=False)
    axes[1].plot(noise_levels, [row["advantage_ratio"]["median"] for row in noise_results], marker="o", color=COLORS["uncertainty"])
    axes[1].fill_between(noise_levels, [row["advantage_ratio"]["q1"] for row in noise_results], [row["advantage_ratio"]["q3"] for row in noise_results], color=COLORS["uncertainty"], alpha=0.16)
    axes[1].axhline(1.0, color="#333333", lw=0.8)
    axes[1].set_xlabel(r"Training-noise $\sigma$")
    axes[1].set_ylabel("Direct / hard-potential MAE")
    axes[1].set_title("(b) Noise mechanism")
    matrix = np.array([[next(row["validation_mae"] for row in ablation if row["hidden_units"] == hidden and row["l2"] == l2) for l2 in l2_values] for hidden in hidden_values])
    image = axes[2].imshow(matrix, aspect="auto", cmap="magma_r")
    axes[2].set_xticks(range(len(l2_values)), ["0", "1e-9", "3e-8", "1e-7"], rotation=30)
    axes[2].set_yticks(range(len(hidden_values)), hidden_values)
    axes[2].set_xlabel(r"$\lambda$")
    axes[2].set_ylabel("Hidden units")
    axes[2].set_title("(c) Validation ablation")
    fig.colorbar(image, ax=axes[2], fraction=0.045, pad=0.03, label="Validation MAE")
    fig.tight_layout()
    save_figure(fig, "fig09_data_efficiency")
    return {
        "efficiency": efficiency,
        "noise_sweep": noise_results,
        "heteroscedastic_advantage_ratio": median_iqr(heteroscedastic_ratios),
        "ablation": ablation,
    }


def thermal_and_infinite_dilution(models):
    x_axis = np.linspace(0.01, 0.99, 120)
    t_axis = np.linspace(292.0, 448.0, 80)
    xx, tt = np.meshgrid(x_axis, t_axis)
    x = np.column_stack([xx.ravel(), 1.0 - xx.ravel()])
    temperature = tt.ravel()
    derivative_model = pair_potential_temperature_derivative(models["potential"], models["spec"], x, temperature)
    delta_t = 1.0e-2
    g_plus = nrtl_predict(x, temperature + delta_t, binary_parameters())[0]
    g_minus = nrtl_predict(x, temperature - delta_t, binary_parameters())[0]
    derivative_reference = (g_plus - g_minus) / (2.0 * delta_t)
    h_reference = -(temperature**2) * derivative_reference
    h_model = -(temperature**2) * derivative_model
    h_error = np.abs(h_model - h_reference)

    temperatures = np.array([300.0, 350.0, 400.0, 440.0])
    epsilon = 1.0e-8
    dilute_x = []
    dilute_t = []
    labels = []
    for value in temperatures:
        dilute_x.extend([[epsilon, 1.0 - epsilon], [1.0 - epsilon, epsilon]])
        dilute_t.extend([value, value])
        labels.extend(["component_1_in_2", "component_2_in_1"])
    dilute_x = np.asarray(dilute_x)
    dilute_t = np.asarray(dilute_t)
    reference = nrtl_predict(dilute_x, dilute_t, binary_parameters())[2]
    potential = pair_potential_predict(models["potential"], models["spec"], dilute_x, dilute_t)[2]
    direct = predict_matched_mlp(models["direct"], dilute_x, dilute_t)[0]
    rk = predict_redlich_kister(models["rk"], dilute_x, dilute_t)[1]
    records = []
    for index, (composition, temp, label) in enumerate(zip(dilute_x, dilute_t, labels)):
        component = 0 if label == "component_1_in_2" else 1
        records.append(
            {
                "temperature_K": float(temp),
                "limit": label,
                "reference_ln_gamma_infinity": float(reference[index, component]),
                "potential_ln_gamma_infinity": float(potential[index, component]),
                "potential_abs_error": float(abs(potential[index, component] - reference[index, component])),
                "direct_abs_error": float(abs(direct[index, component] - reference[index, component])),
                "redlich_kister_abs_error": float(abs(rk[index, component] - reference[index, component])),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    image = axes[0].pcolormesh(x_axis, t_axis, np.log10(h_error.reshape(tt.shape) + 1.0e-8), shading="auto", cmap="viridis")
    axes[0].set_xlabel(r"Mole fraction $x_1$")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(r"(a) $\log_{10}$ error in $H^E/(nR)$")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    for method, marker, color in (("potential_abs_error", "o", COLORS["potential"]), ("direct_abs_error", "s", COLORS["direct"]), ("redlich_kister_abs_error", "^", "#e9c46a")):
        for limit, style in (("component_1_in_2", "-"), ("component_2_in_1", "--")):
            subset = [row for row in records if row["limit"] == limit]
            axes[1].plot([row["temperature_K"] for row in subset], [row[method] for row in subset], marker=marker, linestyle=style, color=color, label=f"{method.split('_')[0]}: {limit.replace('_', ' ')}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel(r"Absolute error in $\ln\gamma_i^\infty$")
    axes[1].set_title("(b) Infinite-dilution limits")
    axes[1].legend(frameon=False, fontsize=6.4, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig10_thermal_limits")
    return {
        "enthalpy_mae_dimensionless_K": float(np.mean(h_error)),
        "enthalpy_max_dimensionless_K": float(np.max(h_error)),
        "infinite_dilution": records,
        "potential_infinite_dilution_mae": float(np.mean([row["potential_abs_error"] for row in records])),
        "direct_infinite_dilution_mae": float(np.mean([row["direct_abs_error"] for row in records])),
        "redlich_kister_infinite_dilution_mae": float(np.mean([row["redlich_kister_abs_error"] for row in records])),
    }


# ---------------------------------------------------------------------------
# Ternary architectural controls, phase classification, and TPD saturation
# ---------------------------------------------------------------------------


def initialize_ordered_pair(hidden_units, embedding_dimension, seed):
    """Initialize a deliberately order-sensitive pair network for ablation."""

    rng = np.random.default_rng(seed)
    feature_dimension = 2 * embedding_dimension + 3
    w = rng.normal(scale=np.sqrt(2.0 / (feature_dimension + hidden_units)), size=(hidden_units, feature_dimension))
    b = rng.normal(scale=0.03, size=hidden_units)
    a = rng.normal(scale=0.05, size=hidden_units)
    return np.concatenate([w.ravel(), b, a, np.zeros(1)])


def unpack_ordered_pair(parameters, hidden_units, embedding_dimension):
    feature_dimension = 2 * embedding_dimension + 3
    cursor = 0
    w = parameters[cursor : cursor + hidden_units * feature_dimension].reshape(hidden_units, feature_dimension)
    cursor += hidden_units * feature_dimension
    b = parameters[cursor : cursor + hidden_units]
    cursor += hidden_units
    a = parameters[cursor : cursor + hidden_units]
    cursor += hidden_units
    return w, b, a, parameters[cursor]


def ordered_pair_g_grad(parameters, embeddings, hidden_units, x, temperature, center=323.15, scale=50.0):
    """Order-sensitive control: concatenated (e_i,e_j,x_i,x_j,T) features."""

    x = anp.asarray(x)
    temperature = anp.asarray(temperature)
    embeddings = anp.asarray(embeddings)
    w, b, a, c = unpack_ordered_pair(parameters, hidden_units, embeddings.shape[1])
    n_samples, n_components = x.shape
    g_excess = anp.zeros(n_samples)
    composition_gradient = anp.zeros((n_samples, n_components))
    t_scaled = (temperature - center) / scale
    for i in range(n_components):
        for j in range(i + 1, n_components):
            xi, xj = x[:, i], x[:, j]
            q = xi * xj
            features = anp.concatenate(
                [
                    anp.tile(embeddings[i], (n_samples, 1)),
                    anp.tile(embeddings[j], (n_samples, 1)),
                    xi[:, None],
                    xj[:, None],
                    t_scaled[:, None],
                ],
                axis=1,
            )
            hidden = anp.tanh(anp.dot(features, w.T) + b)
            psi = anp.dot(hidden, a) + c
            g_excess = g_excess + q * psi
            for m in range(n_components):
                d_q = (xj if m == i else 0.0) + (xi if m == j else 0.0)
                feature_derivative = anp.zeros((n_samples, features.shape[1]))
                if m == i:
                    feature_derivative = feature_derivative + anp.eye(features.shape[1])[2 * embeddings.shape[1]][None, :]
                if m == j:
                    feature_derivative = feature_derivative + anp.eye(features.shape[1])[2 * embeddings.shape[1] + 1][None, :]
                d_hidden = (1.0 - hidden**2) * anp.dot(feature_derivative, w.T)
                d_psi = anp.dot(d_hidden, a)
                basis = anp.eye(n_components)[m]
                composition_gradient = composition_gradient + (d_q * psi + q * d_psi)[:, None] * basis[None, :]
    return g_excess, composition_gradient


def ordered_pair_predict(parameters, embeddings, hidden_units, x, temperature):
    g_excess, gradient = ordered_pair_g_grad(parameters, embeddings, hidden_units, x, temperature)
    ln_gamma = g_excess[:, None] + gradient - anp.sum(x * gradient, axis=1, keepdims=True)
    return np.asarray(g_excess), np.asarray(ln_gamma)


def fit_ordered_pair(embeddings, hidden_units, x, temperature, target, seed, maxiter=750):
    initial = initialize_ordered_pair(hidden_units, embeddings.shape[1], seed)

    def objective(parameters):
        g_excess, gradient = ordered_pair_g_grad(parameters, embeddings, hidden_units, anp.asarray(x), anp.asarray(temperature))
        prediction = g_excess[:, None] + gradient - anp.sum(x * gradient, axis=1, keepdims=True)
        return anp.mean((prediction - target) ** 2) + 3.0e-8 * anp.mean(parameters**2)

    gradient = autograd_grad(objective)
    result = minimize(
        lambda p: float(objective(p)),
        initial,
        jac=lambda p: np.asarray(gradient(p), dtype=float),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1.0e-9, "gtol": 1.0e-6, "maxls": 40},
    )
    return {
        "parameters": np.asarray(result.x),
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "parameter_count": int(len(result.x)),
    }


def tangent_gd_residual_direct(model, x, temperature, step=1.0e-5):
    """Evaluate both independent tangent directions on the ternary simplex."""

    x = np.asarray(x, dtype=float)
    directions = (np.array([1.0, 0.0, -1.0]), np.array([0.0, 1.0, -1.0]))
    residuals = []
    for direction in directions:
        plus = model.predict(np.column_stack([x + step * direction, temperature]))
        minus = model.predict(np.column_stack([x - step * direction, temperature]))
        derivative = (plus - minus) / (2.0 * step)
        residuals.append(np.abs(np.sum(x * derivative, axis=1)))
    return np.column_stack(residuals)


def ternary_model_campaign():
    parameters = ternary_parameters()
    rng = np.random.default_rng(700)
    x_train = composition_samples_ternary(rng, 720)
    x_validation = composition_samples_ternary(np.random.default_rng(701), 240)
    x_test = composition_samples_ternary(np.random.default_rng(702), 1800)
    t_train = np.full(len(x_train), 323.15)
    t_validation = np.full(len(x_validation), 323.15)
    t_test = np.full(len(x_test), 323.15)
    y_train_clean = nrtl_predict(x_train, t_train, parameters)[2]
    y_validation = nrtl_predict(x_validation, t_validation, parameters)[2]
    g_test, _, y_test = nrtl_predict(x_test, t_test, parameters)

    seed_records = []
    potential_models = []
    first_direct = None
    for seed in range(10):
        noise = np.random.default_rng(7100 + seed).normal(scale=0.006, size=y_train_clean.shape)
        noisy = y_train_clean + noise
        spec = PairPotentialSpec(canonical_embeddings(3), hidden_units=20, temperature_center=323.15, temperature_scale=50.0)
        potential = fit_pair_potential(spec, x_train, t_train, noisy, seed=7200 + seed, maxiter=800, l2_weight=3.0e-8)
        g_prediction, _, prediction = pair_potential_predict(potential.parameters, spec, x_test, t_test)
        direct = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(24, 24), activation="tanh", solver="lbfgs", alpha=1.0e-6,
                max_iter=1800, random_state=7300 + seed, tol=1.0e-10,
            ),
        )
        direct.fit(np.column_stack([x_train, t_train]), noisy)
        direct_prediction = direct.predict(np.column_stack([x_test, t_test]))
        seed_records.append(
            {
                "seed": seed,
                "potential_test_mae": float(np.mean(np.abs(prediction - y_test))),
                "potential_g_mae": float(np.mean(np.abs(g_prediction - g_test))),
                "direct_test_mae": float(np.mean(np.abs(direct_prediction - y_test))),
                "potential_converged": potential.converged,
            }
        )
        potential_models.append(potential.parameters)
        if seed == 0:
            first_direct = direct

    primary_parameters = potential_models[0]
    primary_spec = PairPotentialSpec(canonical_embeddings(3), hidden_units=20, temperature_center=323.15, temperature_scale=50.0)
    ordered = fit_ordered_pair(primary_spec.embeddings, 14, x_train, t_train, y_train_clean, seed=7400)
    ordered_prediction = ordered_pair_predict(ordered["parameters"], primary_spec.embeddings, 14, x_test, t_test)[1]

    permutation = np.array([2, 0, 1])
    equivariant_original = pair_potential_predict(primary_parameters, primary_spec, x_test[:300], t_test[:300])[2]
    permuted_spec = PairPotentialSpec(primary_spec.embeddings[permutation], hidden_units=20, temperature_center=323.15, temperature_scale=50.0)
    equivariant_permuted = pair_potential_predict(primary_parameters, permuted_spec, x_test[:300, permutation], t_test[:300])[2]
    ordered_original = ordered_pair_predict(ordered["parameters"], primary_spec.embeddings, 14, x_test[:300], t_test[:300])[1]
    ordered_permuted = ordered_pair_predict(ordered["parameters"], primary_spec.embeddings[permutation], 14, x_test[:300, permutation], t_test[:300])[1]
    direct_original = first_direct.predict(np.column_stack([x_test[:300], t_test[:300]]))
    direct_permuted = first_direct.predict(np.column_stack([x_test[:300, permutation], t_test[:300]]))

    interior = x_test[np.min(x_test, axis=1) > 2.0e-4][:500]
    direct_gd = tangent_gd_residual_direct(first_direct, interior, np.full(len(interior), 323.15))
    potential_values = np.array([row["potential_test_mae"] for row in seed_records])
    direct_values = np.array([row["direct_test_mae"] for row in seed_records])
    test = wilcoxon(potential_values, direct_values, alternative="less", method="auto")

    with (RESULTS / "ternary_seed_replicates.json").open("w", encoding="utf-8") as handle:
        json.dump(seed_records, handle, indent=2, allow_nan=False)
    np.savez_compressed(
        RESULTS / "ternary_revision_models.npz",
        potential_parameters=np.stack(potential_models),
        embeddings=primary_spec.embeddings,
        ordered_parameters=ordered["parameters"],
    )
    return {
        "seed_replicates": seed_records,
        "potential_test_mae": median_iqr(potential_values),
        "direct_test_mae": median_iqr(direct_values),
        "paired_wilcoxon_pvalue": float(test.pvalue),
        "ordered_test_mae": float(np.mean(np.abs(ordered_prediction - y_test))),
        "potential_permutation_max": float(np.max(np.abs(equivariant_permuted - equivariant_original[:, permutation]))),
        "ordered_permutation_max": float(np.max(np.abs(ordered_permuted - ordered_original[:, permutation]))),
        "direct_permutation_max": float(np.max(np.abs(direct_permuted - direct_original[:, permutation]))),
        "direct_tangent_gd_mae": float(np.mean(direct_gd)),
        "ordered_parameter_count": ordered["parameter_count"],
        "potential_parameter_count": primary_spec.parameter_count,
        "direct_parameter_count": int(sum(layer.size for layer in first_direct[-1].coefs_) + sum(layer.size for layer in first_direct[-1].intercepts_)),
    }, {
        "parameters": parameters,
        "spec": primary_spec,
        "potential": primary_parameters,
        "potential_models": potential_models,
        "direct": first_direct,
        "x_test": x_test,
        "t_test": t_test,
        "y_test": y_test,
        "y_validation": y_validation,
    }


def _json_flash(result):
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in result.items()}


def ternary_phase_campaign(models):
    reference_g, model_g, _, model_lng = make_g_functions(models["parameters"], models["spec"], models["potential"])
    rng = np.random.default_rng(7600)
    candidates = np.vstack(
        [
            rng.dirichlet([0.35, 0.35, 0.35], 90),
            rng.dirichlet([3.0, 3.0, 3.0], 70),
            rng.dirichlet([10.0, 1.0, 1.0], 35),
            rng.dirichlet([1.0, 1.0, 10.0], 35),
        ]
    )
    prescreen = [flash_two_phase(feed, reference_g, seed=7700 + index, starts=2, ftol=1.0e-9, maxiter=300) for index, feed in enumerate(candidates)]
    single_indices = [index for index, result in enumerate(prescreen) if result["n_phases"] == 1]
    two_indices = [index for index, result in enumerate(prescreen) if result["n_phases"] == 2]
    if len(single_indices) < 50 or len(two_indices) < 50:
        raise RuntimeError("candidate panel did not contain 50 states of each phase class")
    selected = np.array(single_indices[:50] + two_indices[:50], dtype=int)
    feeds = candidates[selected]

    reference_results = []
    model_results = []
    for index, feed in enumerate(feeds):
        reference_results.append(order_phases(flash_two_phase(feed, reference_g, seed=8000 + index, starts=10, ftol=1.0e-11, maxiter=600)))
        model_results.append(order_phases(flash_two_phase(feed, model_g, seed=9000 + index, starts=10, ftol=1.0e-11, maxiter=600)))
    reference_label = np.array([result["n_phases"] == 2 for result in reference_results], dtype=int)
    model_label = np.array([result["n_phases"] == 2 for result in model_results], dtype=int)

    # Empirical distance to the opposite phase class is a composition-space
    # boundary margin that is defined for both one- and two-phase feeds.
    margin = np.empty(len(feeds))
    for index, feed in enumerate(feeds):
        opposite = feeds[reference_label != reference_label[index]]
        margin[index] = np.min(np.linalg.norm(opposite - feed, axis=1))
    terciles = np.quantile(margin, [1.0 / 3.0, 2.0 / 3.0])
    strata = np.digitize(margin, terciles)
    stratum_rows = []
    for stratum, label in enumerate(("near", "intermediate", "interior")):
        mask = strata == stratum
        correct = int(np.sum(reference_label[mask] == model_label[mask]))
        count = int(np.sum(mask))
        stratum_rows.append({"stratum": label, "count": count, "accuracy": correct / count, "clopper_pearson_95": clopper_pearson(correct, count)})

    tie_errors = []
    beta_errors = []
    for reference, prediction in zip(reference_results, model_results):
        if reference["n_phases"] == 2 and prediction["n_phases"] == 2:
            tie_errors.extend(np.abs(np.r_[reference["phase_a"] - prediction["phase_a"], reference["phase_b"] - prediction["phase_b"]]).tolist())
            beta_errors.append(abs(reference["beta"] - prediction["beta"]))

    correct = int(np.sum(reference_label == model_label))
    matrix = confusion_matrix(reference_label, model_label, labels=[0, 1])

    # Solver saturation is evaluated on three model phases, including the
    # feed with the smallest empirical boundary margin.
    representative_indices = [int(np.argmin(margin)), int(np.argmax(margin * (reference_label == 0))), int(np.argmax(margin * (reference_label == 1)))]
    saturation = []
    for state_index in representative_indices:
        phase = model_results[state_index]["phase_a"]
        for starts in (18, 50, 100):
            for ftol in (1.0e-8, 1.0e-10, 1.0e-12):
                diagnostic = minimum_tpd_diagnostics(phase, model_lng, seed=10000 + state_index + starts, starts=starts, ftol=ftol)
                saturation.append({"state_index": state_index, "starts": starts, "ftol": ftol, **diagnostic})

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    axes[0].imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=11)
    axes[0].set_xticks([0, 1], ["One", "Two"])
    axes[0].set_yticks([0, 1], ["One", "Two"])
    axes[0].set_xlabel("Predicted phases")
    axes[0].set_ylabel("Reference phases")
    axes[0].set_title("(a) Confusion matrix")
    for label_value, marker, color, name in ((0, "o", COLORS["reference"], "One phase"), (1, "^", COLORS["potential"], "Two phases")):
        mask = reference_label == label_value
        error = (reference_label != model_label).astype(float)
        axes[1].scatter(margin[mask], error[mask] + np.random.default_rng(1 + label_value).normal(scale=0.018, size=np.sum(mask)), marker=marker, color=color, alpha=0.68, label=name)
    axes[1].set_xlabel("Distance to opposite-class feed")
    axes[1].set_ylabel("Classification error")
    axes[1].set_yticks([0, 1], ["Correct", "Error"])
    axes[1].set_title("(b) Boundary-stratified errors")
    axes[1].legend(frameon=False)
    for state_index, marker in zip(representative_indices, ("o", "s", "^")):
        subset = [row for row in saturation if row["state_index"] == state_index and row["ftol"] == 1.0e-12]
        axes[2].plot([row["starts"] for row in subset], [abs(row["minimum"]) + 1.0e-16 for row in subset], marker=marker, label=f"State {state_index}")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Random TPD starts")
    axes[2].set_ylabel(r"Absolute minimum TPD")
    axes[2].set_title("(c) Stability-search saturation")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig05_ternary_tielines")

    records = [
        {
            "feed": feed.tolist(),
            "boundary_distance": float(distance),
            "reference": _json_flash(reference),
            "model": _json_flash(prediction),
        }
        for feed, distance, reference, prediction in zip(feeds, margin, reference_results, model_results)
    ]
    with (RESULTS / "ternary_phase_classification.json").open("w", encoding="utf-8") as handle:
        json.dump({"records": records, "tpd_saturation": saturation}, handle, indent=2, allow_nan=False)
    return {
        "feed_count": int(len(feeds)),
        "one_phase_reference": int(np.sum(reference_label == 0)),
        "two_phase_reference": int(np.sum(reference_label == 1)),
        "confusion_matrix_rows_reference": matrix.tolist(),
        "accuracy": correct / len(feeds),
        "accuracy_clopper_pearson_95": clopper_pearson(correct, len(feeds)),
        "boundary_strata": stratum_rows,
        "tie_line_mae": float(np.mean(tie_errors)) if tie_errors else None,
        "beta_mae": float(np.mean(beta_errors)) if beta_errors else None,
        "tpd_saturation": saturation,
    }


# ---------------------------------------------------------------------------
# Ensemble controls, calibration, and controlled process timing
# ---------------------------------------------------------------------------


def _ensemble_fit(binary, *, composition_window, temperature_window, seeds, count=280):
    low_x, high_x = composition_window
    low_t, high_t = temperature_window
    parameters = []
    elapsed = []
    spec = PairPotentialSpec(canonical_embeddings(2), hidden_units=12, temperature_center=370.0, temperature_scale=80.0)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x1 = rng.uniform(low_x, high_x, count)
        x = np.column_stack([x1, 1.0 - x1])
        temperature = rng.uniform(low_t, high_t, count)
        target = nrtl_predict(x, temperature, binary)[2] + rng.normal(scale=0.01, size=(count, 2))
        fit = fit_pair_potential(spec, x, temperature, target, seed=seed + 5000, maxiter=430, l2_weight=5.0e-8)
        parameters.append(fit.parameters)
        elapsed.append(fit.elapsed_seconds)
    return spec, parameters, elapsed


def _ensemble_diagnostics(spec, parameters, x, temperature, target):
    predictions = np.stack([pair_potential_predict(parameter, spec, x, temperature)[2] for parameter in parameters])
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.sqrt(np.mean(np.var(predictions, axis=0, ddof=1), axis=1))
    error = np.mean(np.abs(mean_prediction - target), axis=1)
    return predictions, mean_prediction, uncertainty, error


def uncertainty_campaign():
    binary = binary_parameters()
    spec, parameters, elapsed = _ensemble_fit(binary, composition_window=(0.004, 0.996), temperature_window=(290.0, 370.0), seeds=range(110, 130))
    rng = np.random.default_rng(12000)
    x_in, t_in = composition_samples_binary(rng, 500, 295.0, 365.0)
    x_out, t_out = composition_samples_binary(rng, 500, 390.0, 450.0)
    x = np.vstack([x_in, x_out])
    temperature = np.concatenate([t_in, t_out])
    domain = np.r_[np.zeros(len(x_in), dtype=int), np.ones(len(x_out), dtype=int)]
    target = nrtl_predict(x, temperature, binary)[2]
    predictions, mean_prediction, uncertainty, error = _ensemble_diagnostics(spec, parameters, x, temperature, target)
    distance_temperature = np.maximum.reduce([290.0 - temperature, temperature - 370.0, np.zeros_like(temperature)])

    correlations = {}
    for name, mask in (("pooled", np.ones(len(x), dtype=bool)), ("in_domain", domain == 0), ("temperature_ood", domain == 1)):
        result = spearmanr(uncertainty[mask], error[mask])
        interval, _ = bootstrap_spearman_interval(uncertainty[mask], error[mask], seed=12100 + int(np.sum(mask)), repetitions=1000)
        correlations[name] = {"rho": float(result.statistic), "pvalue": float(result.pvalue), "bootstrap_95": interval}
    temperature_control = spearmanr(distance_temperature, error)
    threshold = float(np.quantile(error, 0.75))
    high_error = (error >= threshold).astype(int)
    auroc_ensemble = float(roc_auc_score(high_error, uncertainty))
    auroc_temperature = float(roc_auc_score(high_error, distance_temperature))

    coverages = np.linspace(1.0, 0.20, 17)
    selective = []
    random_curves = np.empty((200, len(coverages)))
    ordering = np.argsort(uncertainty)
    random_rng = np.random.default_rng(12200)
    for coverage_index, coverage in enumerate(coverages):
        count = max(5, int(round(coverage * len(x))))
        selective.append(float(np.mean(error[ordering[:count]])))
        for repetition in range(200):
            selection = random_rng.choice(len(x), size=count, replace=False)
            random_curves[repetition, coverage_index] = np.mean(error[selection])
    aurc_selective = float(np.trapezoid(selective[::-1], coverages[::-1]) / (coverages.max() - coverages.min()))
    random_mean = np.mean(random_curves, axis=0)
    aurc_random = float(np.trapezoid(random_mean[::-1], coverages[::-1]) / (coverages.max() - coverages.min()))

    # Split-conformal absolute-residual calibration, never reused for ranking.
    x_cal, t_cal = composition_samples_binary(np.random.default_rng(12300), 500, 295.0, 365.0)
    y_cal = nrtl_predict(x_cal, t_cal, binary)[2]
    calibration_predictions = np.mean(np.stack([pair_potential_predict(parameter, spec, x_cal, t_cal)[2] for parameter in parameters]), axis=0)
    calibration_scores = np.max(np.abs(calibration_predictions - y_cal), axis=1)
    quantile_level = np.ceil((len(calibration_scores) + 1) * 0.90) / len(calibration_scores)
    conformal_radius = float(np.quantile(calibration_scores, min(1.0, quantile_level), method="higher"))
    state_max_error = np.max(np.abs(mean_prediction - target), axis=1)
    coverage_in = float(np.mean(state_max_error[domain == 0] <= conformal_radius))
    coverage_out = float(np.mean(state_max_error[domain == 1] <= conformal_radius))

    gp = fit_fixed_gp(x_cal, t_cal, y_cal)
    gp_mean, gp_std = predict_fixed_gp(gp, x, temperature, return_std=True)
    gp_error = np.mean(np.abs(gp_mean - target), axis=1)
    gp_uncertainty = np.mean(gp_std, axis=1) if np.ndim(gp_std) == 2 else np.asarray(gp_std)
    gp_correlation = spearmanr(gp_uncertainty, gp_error)

    composition_spec, composition_parameters, composition_elapsed = _ensemble_fit(
        binary, composition_window=(0.20, 0.80), temperature_window=(290.0, 450.0), seeds=range(210, 230)
    )
    x_center_1 = rng.uniform(0.25, 0.75, 400)
    x_tail_1 = np.r_[rng.uniform(0.004, 0.12, 200), rng.uniform(0.88, 0.996, 200)]
    x_composition = np.column_stack([np.r_[x_center_1, x_tail_1], 1.0 - np.r_[x_center_1, x_tail_1]])
    t_composition = rng.uniform(300.0, 440.0, len(x_composition))
    target_composition = nrtl_predict(x_composition, t_composition, binary)[2]
    _, _, uncertainty_composition, error_composition = _ensemble_diagnostics(composition_spec, composition_parameters, x_composition, t_composition, target_composition)
    composition_domain = np.r_[np.zeros(400, dtype=int), np.ones(400, dtype=int)]
    composition_correlations = {}
    for name, mask in (("pooled", np.ones(800, dtype=bool)), ("central", composition_domain == 0), ("composition_ood", composition_domain == 1)):
        result = spearmanr(uncertainty_composition[mask], error_composition[mask])
        composition_correlations[name] = {"rho": float(result.statistic), "pvalue": float(result.pvalue)}

    fig, axes = plt.subplots(1, 3, figsize=(7.7, 2.8))
    axes[0].scatter(uncertainty[domain == 0], error[domain == 0], s=8, alpha=0.35, color=COLORS["reference"], label="In-domain")
    axes[0].scatter(uncertainty[domain == 1], error[domain == 1], s=9, alpha=0.42, marker="^", color=COLORS["potential"], label="Temperature OOD")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ensemble standard deviation")
    axes[0].set_ylabel("Statewise MAE")
    axes[0].set_title("(a) Stratified ranking")
    axes[0].legend(frameon=False)
    referred = 100.0 * (1.0 - coverages)
    axes[1].plot(referred, selective, marker="o", color=COLORS["uncertainty"], label="Ensemble")
    axes[1].plot(referred, random_mean, color=COLORS["neutral"], label="Random mean")
    axes[1].fill_between(referred, np.quantile(random_curves, 0.025, axis=0), np.quantile(random_curves, 0.975, axis=0), color=COLORS["neutral"], alpha=0.18, label="Random 95% band")
    axes[1].set_xlabel("Referred states (%)")
    axes[1].set_ylabel("Accepted-state MAE")
    axes[1].set_title("(b) Risk--coverage")
    axes[1].legend(frameon=False)
    axes[2].bar([0, 1], [100 * coverage_in, 100 * coverage_out], color=[COLORS["reference"], COLORS["potential"]])
    axes[2].axhline(90, color="#333333", linestyle="--", linewidth=0.9, label="Nominal 90%")
    axes[2].set_xticks([0, 1], ["In-domain", "Temp. OOD"])
    axes[2].set_ylabel("Simultaneous coverage (%)")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("(c) Split-conformal audit")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig06_uncertainty")

    np.savez_compressed(
        RESULTS / "uncertainty_revision.npz", parameters=np.stack(parameters), x=x, temperature=temperature,
        domain=domain, target=target, prediction=mean_prediction, uncertainty=uncertainty, error=error,
    )
    return {
        "ensemble_size": len(parameters),
        "correlations": correlations,
        "temperature_distance_control": {"rho": float(temperature_control.statistic), "pvalue": float(temperature_control.pvalue)},
        "high_error_threshold": threshold,
        "auroc_ensemble": auroc_ensemble,
        "auroc_temperature_distance": auroc_temperature,
        "coverage_grid": coverages.tolist(),
        "selective_risk": selective,
        "random_risk_mean": random_mean.tolist(),
        "random_risk_95": [np.quantile(random_curves, 0.025, axis=0).tolist(), np.quantile(random_curves, 0.975, axis=0).tolist()],
        "aurc_selective": aurc_selective,
        "aurc_random": aurc_random,
        "conformal_nominal": 0.90,
        "conformal_radius": conformal_radius,
        "conformal_coverage_in_domain": coverage_in,
        "conformal_coverage_temperature_ood": coverage_out,
        "gp_uncertainty_rho": float(gp_correlation.statistic),
        "gp_uncertainty_pvalue": float(gp_correlation.pvalue),
        "composition_shift_correlations": composition_correlations,
        "fit_seconds": median_iqr(elapsed),
        "composition_fit_seconds": median_iqr(composition_elapsed),
    }


def hardware_manifest():
    cpu_model = platform.processor() or "not reported by platform"
    cpu_count = os.cpu_count()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        np.show_config()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpu_count": cpu_count,
        "thread_environment": {key: os.environ.get(key, "unset") for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "numpy_configuration": configuration.getvalue(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "scikit-learn", "matplotlib", "autograd")
        },
    }


def process_and_timing_campaign(models):
    reference_g, model_g, _, _ = make_g_functions(models["parameters"], models["spec"], models["potential"])
    feed = np.array([0.46, 0.10, 0.44])

    # Three discarded warm-up calls per evaluator; then 30 interleaved repeats.
    for warmup in range(3):
        flash_two_phase(feed, reference_g, seed=13000 + warmup, starts=6, ftol=1.0e-10, maxiter=500)
        flash_two_phase(feed, model_g, seed=13100 + warmup, starts=6, ftol=1.0e-10, maxiter=500)
    timings = {"reference": [], "potential": []}
    diagnostics = {"reference": [], "potential": []}
    for repetition in range(30):
        for name, function, seed_offset in (("reference", reference_g, 14000), ("potential", model_g, 15000)):
            started = perf_counter()
            result = flash_two_phase(feed, function, seed=seed_offset + repetition, starts=6, ftol=1.0e-10, maxiter=500)
            timings[name].append(perf_counter() - started)
            diagnostics[name].append({"function_evaluations": result["total_function_evaluations"], "iterations": result["total_iterations"], "successful_starts": result["successful_starts"]})

    def run_cascade(g_function, solvent_ratio, stages, seed_offset):
        inventory = np.array([0.85, 0.15, 1.0e-8])
        initial_solute = inventory[1]
        balance_error = []
        total_evaluations = 0
        for stage in range(stages):
            mixed = inventory + np.array([1.0e-8, 1.0e-8, solvent_ratio])
            total = mixed.sum()
            result = order_phases(flash_two_phase(mixed / total, g_function, seed=seed_offset + stage, starts=8, ftol=1.0e-11, maxiter=600))
            total_evaluations += result["total_function_evaluations"]
            if result["n_phases"] == 1:
                raffinate_moles, raffinate_x, reconstructed = total, mixed / total, mixed
            else:
                raffinate_moles = result["beta"] * total
                extract_moles = (1.0 - result["beta"]) * total
                raffinate_x = np.asarray(result["phase_a"])
                reconstructed = raffinate_moles * raffinate_x + extract_moles * np.asarray(result["phase_b"])
            balance_error.append(float(np.max(np.abs(reconstructed - mixed))))
            inventory = raffinate_moles * raffinate_x
        return {
            "recovery": float(1.0 - inventory[1] / initial_solute),
            "max_balance_error": float(max(balance_error)),
            "function_evaluations": int(total_evaluations),
        }

    cascade = []
    for stages in (4, 8, 12):
        for ratio in (0.25, 0.50, 1.00):
            reference = run_cascade(reference_g, ratio, stages, 16000 + 100 * stages + int(10 * ratio))
            prediction = run_cascade(model_g, ratio, stages, 17000 + 100 * stages + int(10 * ratio))
            cascade.append({"stages": stages, "solvent_to_feed": ratio, "reference": reference, "potential": prediction, "absolute_recovery_error": abs(reference["recovery"] - prediction["recovery"])})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axes[0].boxplot([timings["reference"], timings["potential"]], tick_labels=["NRTL", "Potential"], showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall time per flash (s)")
    axes[0].set_title("(a) 30 post-warm-up repeats")
    for stages, marker in ((4, "o"), (8, "s"), (12, "^")):
        subset = [row for row in cascade if row["stages"] == stages]
        axes[1].plot([row["solvent_to_feed"] for row in subset], [100 * row["absolute_recovery_error"] for row in subset], marker=marker, label=f"{stages} stages")
    axes[1].set_xlabel("Fresh-solvent/feed molar ratio per stage")
    axes[1].set_ylabel("Absolute recovery error (percentage points)")
    axes[1].set_title("(b) Cascade stress test")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18)
    fig.tight_layout()
    save_figure(fig, "fig07_extraction_cascade")

    return {
        "warmup_repeats": 3,
        "timed_repeats": 30,
        "flash_settings": {"random_starts": 6, "deterministic_starts": 4, "ftol": 1.0e-10, "maxiter": 500},
        "reference_seconds": median_iqr(timings["reference"]),
        "potential_seconds": median_iqr(timings["potential"]),
        "median_runtime_ratio": float(np.median(timings["potential"]) / np.median(timings["reference"])),
        "reference_function_evaluations": median_iqr([row["function_evaluations"] for row in diagnostics["reference"]]),
        "potential_function_evaluations": median_iqr([row["function_evaluations"] for row in diagnostics["potential"]]),
        "reference_iterations": median_iqr([row["iterations"] for row in diagnostics["reference"]]),
        "potential_iterations": median_iqr([row["iterations"] for row in diagnostics["potential"]]),
        "cascade": cascade,
    }


def _sci(value, digits=2):
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"


def write_revision_macros(metrics):
    baseline = metrics["binary_baselines"]
    gd = metrics["gibbs_duhem_refinement"]
    binodal = metrics["binodal_refinement"]
    thermal = metrics["thermal_limits"]
    ternary = metrics["ternary_models"]
    phase = metrics["ternary_phase"]
    uncertainty = metrics["uncertainty"]
    timing = metrics["process_timing"]
    lines = [
        "% Auto-generated by experiments/run_major_revision.py; do not edit.",
        rf"\newcommand{{\SeedCount}}{{{len(baseline['replicates'])}}}",
        rf"\newcommand{{\PotentialIdentifiableParameters}}{{{baseline['parameter_counts']['potential_identifiable_binary']}}}",
        rf"\newcommand{{\PotentialStoredParameters}}{{{baseline['parameter_counts']['potential_stored']}}}",
        rf"\newcommand{{\PotentialMedianMAE}}{{${_sci(baseline['summary']['potential']['median'])}$}}",
        rf"\newcommand{{\PotentialMAEIQR}}{{${_sci(baseline['summary']['potential']['q1'])}$--${_sci(baseline['summary']['potential']['q3'])}$}}",
        rf"\newcommand{{\DirectMatchedMedianMAE}}{{${_sci(baseline['summary']['direct_matched']['median'])}$}}",
        rf"\newcommand{{\SoftGDMedianMAE}}{{${_sci(baseline['summary']['soft_gd']['median'])}$}}",
        rf"\newcommand{{\RedlichKisterMedianMAE}}{{${_sci(baseline['summary']['redlich_kister']['median'])}$}}",
        rf"\newcommand{{\NRTLFixedMedianMAE}}{{${_sci(baseline['summary']['nrtl_fixed_alpha']['median'])}$}}",
        rf"\newcommand{{\NRTLFreeMedianMAE}}{{${_sci(baseline['summary']['nrtl_free_alpha']['median'])}$}}",
        rf"\newcommand{{\GPPMedianMAE}}{{${_sci(baseline['summary']['gaussian_process']['median'])}$}}",
        rf"\newcommand{{\PotentialVsDirectP}}{{${_sci(baseline['paired_tests']['direct_matched']['pvalue'])}$}}",
        rf"\newcommand{{\PotentialVsRKP}}{{{baseline['paired_tests']['redlich_kister']['pvalue']:.3f}}}",
        rf"\newcommand{{\GDHardSlope}}{{{gd['second_order_slopes']['potential']:.2f}}}",
        rf"\newcommand{{\GDDirectFine}}{{${_sci(gd['rows'][-1]['direct'])}$}}",
        rf"\newcommand{{\GDHardFine}}{{${_sci(gd['rows'][-1]['potential'])}$}}",
        rf"\newcommand{{\BinodalFineMAE}}{{${_sci(binodal['fine_grid_endpoint_mae'])}$}}",
        rf"\newcommand{{\BinodalFineMax}}{{${_sci(binodal['fine_grid_composition_hausdorff_max'])}$}}",
        rf"\newcommand{{\BinodalTemperatureMAE}}{{{binodal['inverse_temperature_mae_K']:.2f}\,K}}",
        rf"\newcommand{{\EnthalpyMAE}}{{{thermal['enthalpy_mae_dimensionless_K']:.2f}\,K}}",
        rf"\newcommand{{\EnthalpyMax}}{{{thermal['enthalpy_max_dimensionless_K']:.2f}\,K}}",
        rf"\newcommand{{\InfiniteDilutionMAE}}{{${_sci(thermal['potential_infinite_dilution_mae'])}$}}",
        rf"\newcommand{{\InfiniteDilutionDirectMAE}}{{${_sci(thermal['direct_infinite_dilution_mae'])}$}}",
        rf"\newcommand{{\InfiniteDilutionRKMAE}}{{${_sci(thermal['redlich_kister_infinite_dilution_mae'])}$}}",
        rf"\newcommand{{\TernaryPotentialMAE}}{{${_sci(ternary['potential_test_mae']['median'])}$}}",
        rf"\newcommand{{\TernaryDirectMAE}}{{${_sci(ternary['direct_test_mae']['median'])}$}}",
        rf"\newcommand{{\TernaryOrderedMAE}}{{${_sci(ternary['ordered_test_mae'])}$}}",
        rf"\newcommand{{\TernaryDirectGDP}}{{${_sci(ternary['direct_tangent_gd_mae'])}$}}",
        rf"\newcommand{{\PermutationHard}}{{${_sci(ternary['potential_permutation_max'])}$}}",
        rf"\newcommand{{\PermutationOrdered}}{{${_sci(ternary['ordered_permutation_max'])}$}}",
        rf"\newcommand{{\PhaseFeedCount}}{{{phase['feed_count']}}}",
        rf"\newcommand{{\PhaseAccuracyRevised}}{{{100 * phase['accuracy']:.1f}\%}}",
        rf"\newcommand{{\PhaseAccuracyCILow}}{{{100 * phase['accuracy_clopper_pearson_95'][0]:.1f}\%}}",
        rf"\newcommand{{\PhaseAccuracyCIHigh}}{{{100 * phase['accuracy_clopper_pearson_95'][1]:.1f}\%}}",
        rf"\newcommand{{\PhaseConfusionTN}}{{{phase['confusion_matrix_rows_reference'][0][0]}}}",
        rf"\newcommand{{\PhaseConfusionFP}}{{{phase['confusion_matrix_rows_reference'][0][1]}}}",
        rf"\newcommand{{\PhaseConfusionFN}}{{{phase['confusion_matrix_rows_reference'][1][0]}}}",
        rf"\newcommand{{\PhaseConfusionTP}}{{{phase['confusion_matrix_rows_reference'][1][1]}}}",
        rf"\newcommand{{\PhaseTieMAE}}{{${_sci(phase['tie_line_mae'])}$}}",
        rf"\newcommand{{\PhaseBetaMAE}}{{${_sci(phase['beta_mae'])}$}}",
        rf"\newcommand{{\EnsembleSizeRevised}}{{{uncertainty['ensemble_size']}}}",
        rf"\newcommand{{\SpearmanPooledRevised}}{{{uncertainty['correlations']['pooled']['rho']:.3f}}}",
        rf"\newcommand{{\SpearmanPooledCILow}}{{{uncertainty['correlations']['pooled']['bootstrap_95'][0]:.3f}}}",
        rf"\newcommand{{\SpearmanPooledCIHigh}}{{{uncertainty['correlations']['pooled']['bootstrap_95'][1]:.3f}}}",
        rf"\newcommand{{\SpearmanInRevised}}{{{uncertainty['correlations']['in_domain']['rho']:.3f}}}",
        rf"\newcommand{{\SpearmanOODRevised}}{{{uncertainty['correlations']['temperature_ood']['rho']:.3f}}}",
        rf"\newcommand{{\TemperatureControlRho}}{{{uncertainty['temperature_distance_control']['rho']:.3f}}}",
        rf"\newcommand{{\EnsembleAUROC}}{{{uncertainty['auroc_ensemble']:.3f}}}",
        rf"\newcommand{{\TemperatureAUROC}}{{{uncertainty['auroc_temperature_distance']:.3f}}}",
        rf"\newcommand{{\SelectiveAURC}}{{{uncertainty['aurc_selective']:.4f}}}",
        rf"\newcommand{{\RandomAURC}}{{{uncertainty['aurc_random']:.4f}}}",
        rf"\newcommand{{\CompositionPooledRho}}{{{uncertainty['composition_shift_correlations']['pooled']['rho']:.3f}}}",
        rf"\newcommand{{\CompositionInRho}}{{{uncertainty['composition_shift_correlations']['central']['rho']:.3f}}}",
        rf"\newcommand{{\CompositionOODRho}}{{{uncertainty['composition_shift_correlations']['composition_ood']['rho']:.3f}}}",
        rf"\newcommand{{\GPRho}}{{{uncertainty['gp_uncertainty_rho']:.3f}}}",
        rf"\newcommand{{\ConformalCoverageIn}}{{{100 * uncertainty['conformal_coverage_in_domain']:.1f}\%}}",
        rf"\newcommand{{\ConformalCoverageOOD}}{{{100 * uncertainty['conformal_coverage_temperature_ood']:.1f}\%}}",
        rf"\newcommand{{\TimingRepeats}}{{{timing['timed_repeats']}}}",
        rf"\newcommand{{\RuntimeRatioRevised}}{{{timing['median_runtime_ratio']:.2f}}}",
        rf"\newcommand{{\ReferenceTimingMedian}}{{{timing['reference_seconds']['median']:.3f}\,s}}",
        rf"\newcommand{{\ReferenceTimingIQR}}{{{timing['reference_seconds']['q1']:.3f}--{timing['reference_seconds']['q3']:.3f}\,s}}",
        rf"\newcommand{{\PotentialTimingMedian}}{{{timing['potential_seconds']['median']:.3f}\,s}}",
        rf"\newcommand{{\PotentialTimingIQR}}{{{timing['potential_seconds']['q1']:.3f}--{timing['potential_seconds']['q3']:.3f}\,s}}",
        rf"\newcommand{{\ReferenceEvalsMedian}}{{{timing['reference_function_evaluations']['median']:.0f}}}",
        rf"\newcommand{{\PotentialEvalsMedian}}{{{timing['potential_function_evaluations']['median']:.0f}}}",
        rf"\newcommand{{\CascadeWorstPP}}{{{100 * max(row['absolute_recovery_error'] for row in timing['cascade']):.3f}}}",
    ]
    (SECTIONS / "revision_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    started = perf_counter()
    baseline_metrics, binary_models = binary_baseline_campaign()
    uniquac_metrics = uniquac_supplemental_campaign()
    baseline_property_metrics = baseline_property_campaign(binary_models)
    binary_residual_figure(binary_models)
    gd_metrics = gibbs_duhem_refinement(binary_models)
    binodal_metrics = binodal_refinement(binary_models)
    efficiency_metrics = data_efficiency_noise_and_ablation(binary_models, baseline_metrics["selected_l2"])
    thermal_metrics = thermal_and_infinite_dilution(binary_models)
    ternary_metrics, ternary_models = ternary_model_campaign()
    phase_metrics = ternary_phase_campaign(ternary_models)
    uncertainty_metrics = uncertainty_campaign()
    timing_metrics = process_and_timing_campaign(ternary_models)
    metrics = {
        "epistemic_label": "E7 -- fully synthetic NRTL verification; no experimental molecular validation is claimed",
        "binary_baselines": baseline_metrics,
        "uniquac_baseline": uniquac_metrics,
        "baseline_properties": baseline_property_metrics,
        "gibbs_duhem_refinement": gd_metrics,
        "binodal_refinement": binodal_metrics,
        "data_efficiency_noise_ablation": efficiency_metrics,
        "thermal_limits": thermal_metrics,
        "ternary_models": ternary_metrics,
        "ternary_phase": phase_metrics,
        "uncertainty": uncertainty_metrics,
        "process_timing": timing_metrics,
        "hardware": hardware_manifest(),
        "runtime_seconds_total": float(perf_counter() - started),
    }
    with (RESULTS / "revision_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    write_revision_macros(metrics)
    print(json.dumps({"status": "complete", "runtime_seconds_total": metrics["runtime_seconds_total"]}, indent=2))


if __name__ == "__main__":
    main()
