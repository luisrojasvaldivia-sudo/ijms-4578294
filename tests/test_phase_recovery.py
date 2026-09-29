"""Six regression checks for trace-feed feasibility and the corrective rerun."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase_recovery import allocation_energy_gradient, decode_allocation
from src.thermo import nrtl_g_grad_lngamma, pair_potential_predict, PairPotentialSpec
from experiments.run_all import ternary_parameters
from experiments.recover_phase_failure import energy_function


def main():
    recovery = json.loads((ROOT / "results/phase_failure_recovery.json").read_text())
    row = recovery["recovered_records"][0]
    feed = np.asarray(row["feed"])
    archive = np.load(ROOT / "results/ternary_revision_models.npz")
    spec = PairPotentialSpec(archive["embeddings"], hidden_units=20,
                             temperature_center=323.15, temperature_scale=50.)
    functions = {
        "reference": energy_function(lambda x: nrtl_g_grad_lngamma(x, 323.15, *ternary_parameters())),
        "model": energy_function(lambda x: pair_potential_predict(archive["potential_parameters"][0], spec, x, 323.15))}
    # 1. Exact original infeasibility and preservation of inputs, not relabeling.
    assert feed[1] < 2e-6 and np.flatnonzero(feed < 2e-6).tolist() == [1]
    assert hashlib.sha256((ROOT / "results/ternary_phase_classification.json").read_bytes()).hexdigest() == recovery["source_sha256"]
    assert hashlib.sha256((ROOT / "results/ternary_revision_models.npz").read_bytes()).hexdigest() == recovery["model_sha256"]
    # 2. Feasible parametrization over 100 trace-feed allocations.
    for r in np.random.default_rng(47).uniform(.001,.999,(100,3)):
        beta,a,b = decode_allocation(feed,r)
        assert min(a.min(),b.min()) > 0
        np.testing.assert_allclose(beta*a+(1-beta)*b,feed,rtol=1e-14,atol=1e-16)
    # 3. Analytic extensive-energy gradient checked independently.
    point = np.array([.8,.4,.2])
    for f in functions.values():
        _, derivative = allocation_energy_gradient(feed, point, f)
        h = 1e-5
        numeric = np.array([(allocation_energy_gradient(feed,point+h*np.eye(3)[i],f)[0]-allocation_energy_gradient(feed,point-h*np.eye(3)[i],f)[0])/(2*h) for i in range(3)])
        np.testing.assert_allclose(derivative,numeric,atol=2e-9,rtol=2e-6)
    # 4. Independent recomputation of recovered balances, energy and mu.
    for name, f in functions.items():
        solution = row["solutions"][name]
        a,b = np.array(solution["phase_a"]),np.array(solution["phase_b"])
        beta = solution["beta"]
        ga,ma = f(a)
        gb,mb = f(b)
        assert np.max(np.abs(ma-mb)) < 1e-8
        np.testing.assert_allclose(beta*a+(1-beta)*b,feed,atol=1e-15,rtol=0)
        assert abs(beta*ga+(1-beta)*gb-solution["objective"]) < 1e-13
        assert f(feed)[0]-solution["objective"] > .35
        assert solution["accepted_two_phase_count"] >= 50
    # 5. All declared TPD checks passed, with no global-certificate claim.
    for solution in row["solutions"].values():
        assert len(solution["tpd_checks"]) == 6
        assert all(r["minimum_all_observed"] > -1e-8 and not r["global_certificate"]
                   and r["failed_starts"] == 0 for r in solution["tpd_checks"])
    # 6. Corrected 100/100 convergence and preservation of other 99 states.
    original = json.loads((ROOT / "results/ternary_phase_classification.json").read_text())["records"]
    corrected = json.loads((ROOT / "results/ternary_phase_classification_corrected.json").read_text())["records"]
    for i,(a,b) in enumerate(zip(original,corrected)):
        assert a["feed"] == b["feed"]
        if i != 47:
            assert a["reference"] == b["reference"] and a["model"] == b["model"]
    assert all(r["reference"]["success"] and r["model"]["success"] for r in corrected)
    assert sum(r["reference"]["n_phases"] == r["model"]["n_phases"] for r in corrected) == 98
    assert recovery["corrected_panel"]["confusion_matrix_rows_reference_1_2"] == [[47,0],[2,51]]
    print("6 trace-feed recovery regression checks passed")


if __name__ == "__main__":
    main()
