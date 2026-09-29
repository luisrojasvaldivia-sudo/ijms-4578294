"""Minimal structural tests independent of fitted numerical values."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thermo import (  # noqa: E402
    PairPotentialSpec,
    canonical_embeddings,
    initialize_pair_potential,
    ln_gamma_from_potential,
    nrtl_g_grad_lngamma,
    pair_potential_g_grad,
    pair_potential_predict,
    pair_potential_temperature_derivative,
)


def test_nrtl_gibbs_duhem_on_binary_path():
    x1 = np.linspace(0.01, 0.99, 2001)
    x = np.column_stack([x1, 1.0 - x1])
    t = np.full(len(x), 340.0)
    tau0 = np.zeros((2, 2))
    tau_t = np.array([[0.0, 850.0], [850.0, 0.0]])
    alpha = np.array([[0.0, 0.2], [0.2, 0.0]])
    _, _, ln_gamma = nrtl_g_grad_lngamma(x, t, tau0, tau_t, alpha)
    derivative = np.gradient(ln_gamma, x1, axis=0, edge_order=2)
    residual = np.abs(np.sum(x * derivative, axis=1))
    assert float(np.mean(residual[10:-10])) < 2.0e-5


def test_pair_potential_has_exact_resident_pure_limit():
    spec = PairPotentialSpec(canonical_embeddings(3), hidden_units=5)
    parameters = initialize_pair_potential(spec, seed=2)
    x = np.eye(3)
    t = np.full(3, 323.15)
    g, composition_gradient = pair_potential_g_grad(parameters, spec, x, t)
    ln_gamma = ln_gamma_from_potential(g, composition_gradient, x)
    assert np.max(np.abs(np.diag(np.asarray(ln_gamma)))) < 1.0e-12
    assert np.max(np.abs(np.asarray(g))) < 1.0e-12


def test_pair_potential_is_permutation_equivariant():
    rng = np.random.default_rng(12)
    spec = PairPotentialSpec(canonical_embeddings(3), hidden_units=6)
    parameters = initialize_pair_potential(spec, seed=4)
    x = rng.dirichlet(np.ones(3), size=30)
    t = np.full(30, 323.15)
    g, gradient = pair_potential_g_grad(parameters, spec, x, t)
    ln_gamma = np.asarray(ln_gamma_from_potential(g, gradient, x))
    permutation = np.array([2, 0, 1])
    spec_permuted = PairPotentialSpec(spec.embeddings[permutation], hidden_units=6)
    g_p, gradient_p = pair_potential_g_grad(parameters, spec_permuted, x[:, permutation], t)
    ln_gamma_p = np.asarray(ln_gamma_from_potential(g_p, gradient_p, x[:, permutation]))
    assert np.max(np.abs(ln_gamma_p - ln_gamma[:, permutation])) < 1.0e-11


def test_analytic_temperature_derivative_matches_centered_difference():
    rng = np.random.default_rng(33)
    spec = PairPotentialSpec(
        canonical_embeddings(3), hidden_units=7, temperature_center=323.15, temperature_scale=50.0
    )
    parameters = initialize_pair_potential(spec, seed=34)
    x = rng.dirichlet(np.ones(3), size=24)
    temperature = rng.uniform(295.0, 430.0, size=len(x))
    step = 1.0e-3
    analytic = pair_potential_temperature_derivative(parameters, spec, x, temperature)
    plus = pair_potential_predict(parameters, spec, x, temperature + step)[0]
    minus = pair_potential_predict(parameters, spec, x, temperature - step)[0]
    numeric = (plus - minus) / (2.0 * step)
    assert np.max(np.abs(analytic - numeric)) < 2.0e-10


def test_nrtl_infinite_dilution_reference_values():
    temperatures = np.array([300.0, 350.0, 400.0, 440.0])
    expected = np.array([4.44100539, 3.92276838, 3.51426079, 3.24453380])
    epsilon = 1.0e-10
    x = np.column_stack([np.full(len(temperatures), epsilon), np.full(len(temperatures), 1.0 - epsilon)])
    tau0 = np.zeros((2, 2))
    tau_t = np.array([[0.0, 850.0], [850.0, 0.0]])
    alpha = np.array([[0.0, 0.2], [0.2, 0.0]])
    _, _, ln_gamma = nrtl_g_grad_lngamma(x, temperatures, tau0, tau_t, alpha)
    assert np.max(np.abs(ln_gamma[:, 0] - expected)) < 1.0e-8
    assert np.max(np.abs(ln_gamma[:, 1])) < 1.0e-8


if __name__ == "__main__":
    test_nrtl_gibbs_duhem_on_binary_path()
    test_pair_potential_has_exact_resident_pure_limit()
    test_pair_potential_is_permutation_equivariant()
    test_analytic_temperature_derivative_matches_centered_difference()
    test_nrtl_infinite_dilution_reference_values()
    print("5 structural and derivative tests passed")
