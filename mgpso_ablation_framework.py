from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import numpy as np
from scipy import stats

try:
    from pymoo.problems import get_problem
    from pymoo.indicators.igd import IGD as PymooIGD
    from pymoo.indicators.hv import HV as PymooHV
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False


#Ablation Configurations & Hyperparameters

@dataclass(frozen=True)
class AblationConfig:
    name: str
    ring: bool
    ts: bool
    pm: bool
    scalarization: str = "sum" 


ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    "PROPOSED":  AblationConfig("PROPOSED",  ring=True,  ts=True,  pm=True),
    "BASELINE":  AblationConfig("BASELINE",  ring=False, ts=False, pm=False),
    "NO_PM":     AblationConfig("NO_PM",     ring=True,  ts=True,  pm=False),
    "NO_RING":   AblationConfig("NO_RING",   ring=False, ts=True,  pm=True),
    "NO_TS":     AblationConfig("NO_TS",     ring=True,  ts=False, pm=True),
    "ONLY_PM":   AblationConfig("ONLY_PM",   ring=False, ts=False, pm=True),
    "ONLY_RING": AblationConfig("ONLY_RING", ring=True,  ts=False, pm=False),
    "ONLY_TS":   AblationConfig("ONLY_TS",   ring=False, ts=True,  pm=False),
}


@dataclass
class HyperParameters:
    swarm_size: int = 150
    archive_size: int = 300
    w: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    c3: float = 1.49618
    lam: float = 0.5
    t_size: int = 5
    eta: float = 20.0
    r_ring: int = 3
    vmax_ratio: float = 0.2


# Vectorized Dominance & NSGA-II Crowding Distance

def find_non_dominated_indices(F: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated rows in objective matrix F (minimization)."""
    K = F.shape[0]
    if K == 0:
        return np.array([], dtype=int)
    leq = np.all(F[None, :, :] <= F[:, None, :], axis=2)
    lt = np.any(F[None, :, :] < F[:, None, :], axis=2)
    return np.where(~np.any(leq & lt, axis=1))[0]


def compute_crowding_distance(F: np.ndarray) -> np.ndarray:
    """Compute NSGA-II crowding distance CD_j for archive objective matrix F."""
    K, M = F.shape
    if K <= 2:
        return np.full(K, np.inf, dtype=float)
    cd = np.zeros(K, dtype=float)
    for m in range(M):
        order = np.argsort(F[:, m])
        f_sorted = F[order, m]
        f_range = f_sorted[-1] - f_sorted[0]
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        if f_range > 1e-12:
            cd[order[1:-1]] += (f_sorted[2:] - f_sorted[:-2]) / f_range
    return cd


# Unified-Swarm Multi-Guide PSO (Algorithm 1 + Tchebycheff Control)

def run_mgpso_ablation(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    xl: np.ndarray,
    xu: np.ndarray,
    n_gen: int,
    config: AblationConfig,
    hp: HyperParameters = HyperParameters(),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute Algorithm 1 (Unified-Swarm Multi-Guide PSO for 2^3 Factorial Ablation)."""
    rng = np.random.default_rng(seed)
    N, A_max, d = hp.swarm_size, hp.archive_size, len(xl)
    pm_prob = 1.0 / d
    bounds_range = xu - xl
    v_max = hp.vmax_ratio * bounds_range

    X = rng.uniform(xl, xu, size=(N, d))
    V = rng.uniform(-v_max, v_max, size=(N, d))
    F = evaluate_fn(X)
    P, P_F = X.copy(), F.copy()

    nd_idx = find_non_dominated_indices(P_F)
    archive_X, archive_F = P[nd_idx].copy(), P_F[nd_idx].copy()

    offsets = np.arange(-hp.r_ring, hp.r_ring + 1)
    ring_neighbors = (np.arange(N)[:, None] + offsets[None, :]) % N

    for _ in range(n_gen):
        cd = compute_crowding_distance(archive_F)
        arch_len = len(archive_X)

        # Step 1: Social neighborhood guide (sum scalarization vs Tchebycheff control)
        if config.scalarization == "chebyshev":
            agg_pbest = np.max(P_F, axis=1)
        else:
            agg_pbest = np.sum(P_F, axis=1)

        if config.ring:
            best_local_pos = np.argmin(agg_pbest[ring_neighbors], axis=1)
            Y_hat = P[ring_neighbors[np.arange(N), best_local_pos]]
        else:
            Y_hat = np.broadcast_to(P[int(np.argmin(agg_pbest))], (N, d)).copy()

        # Step 2: External archive guide
        if config.ts and arch_len > 1:
            t_eff = min(hp.t_size, arch_len)
            tour_idx = rng.integers(0, arch_len, size=(N, t_eff))
            winner_pos = np.argmax(cd[tour_idx], axis=1)
            A_hat = archive_X[tour_idx[np.arange(N), winner_pos]]
        else:
            A_hat = archive_X[rng.integers(0, arch_len, size=N)]

        # Step 3: Velocity & Position update
        r1, r2, r3 = rng.random((N, d)), rng.random((N, d)), rng.random((N, d))
        V = (
            hp.w * V
            + hp.c1 * r1 * (P - X)
            + hp.lam * hp.c2 * r2 * (Y_hat - X)
            + (1.0 - hp.lam) * hp.c3 * r3 * (A_hat - X)
        )
        V = np.clip(V, -v_max, v_max)
        X = X + V

        # Step 4: Post-update Polynomial Mutation (PM)
        if config.pm:
            mut_mask = rng.random((N, d)) < pm_prob
            u = rng.random((N, d))
            delta_q = np.where(
                u <= 0.5,
                np.power(2.0 * u, 1.0 / (hp.eta + 1.0)) - 1.0,
                1.0 - np.power(2.0 * (1.0 - u), 1.0 / (hp.eta + 1.0)),
            )
            X = np.where(mut_mask, X + delta_q * bounds_range, X)

        X = np.clip(X, xl, xu)
        F = evaluate_fn(X)

        # Step 5: Personal best update (if p_i does not strictly dominate x_i)
        p_dominates_x = np.all(P_F <= F, axis=1) & np.any(P_F < F, axis=1)
        P[~p_dominates_x] = X[~p_dominates_x]
        P_F[~p_dominates_x] = F[~p_dominates_x]

        # Archive update & crowding-distance truncation
        cand_X = np.vstack([archive_X, X])
        cand_F = np.vstack([archive_F, F])
        _, unique_idx = np.unique(np.round(cand_F, decimals=12), axis=0, return_index=True)
        cand_X, cand_F = cand_X[np.sort(unique_idx)], cand_F[np.sort(unique_idx)]
        nd_idx = find_non_dominated_indices(cand_F)
        archive_X, archive_F = cand_X[nd_idx], cand_F[nd_idx]

        while len(archive_X) > A_max:
            remove_idx = int(np.argmin(compute_crowding_distance(archive_F)))
            archive_X = np.delete(archive_X, remove_idx, axis=0)
            archive_F = np.delete(archive_F, remove_idx, axis=0)

    return archive_X, archive_F


# Baselines: Canonical Multi-Swarm MGPSO & Standard SMPSO

def run_canonical_multiswarm_mgpso(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    xl: np.ndarray,
    xu: np.ndarray,
    n_gen: int,
    n_obj: int = 2,
    hp: HyperParameters = HyperParameters(),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical Multi-Swarm MGPSO (Scheepers et al., 2019) with M objective-specific sub-swarms."""
    rng = np.random.default_rng(seed)
    N, A_max, d = hp.swarm_size, hp.archive_size, len(xl)
    v_max = hp.vmax_ratio * (xu - xl)

    X = rng.uniform(xl, xu, size=(N, d))
    V = rng.uniform(-v_max, v_max, size=(N, d))
    F = evaluate_fn(X)
    P, P_F = X.copy(), F.copy()

    nd_idx = find_non_dominated_indices(P_F)
    archive_X, archive_F = P[nd_idx].copy(), P_F[nd_idx].copy()
    sub_splits = np.array_split(np.arange(N), n_obj)

    for _ in range(n_gen):
        cd = compute_crowding_distance(archive_F)
        arch_len = len(archive_X)
        Y_hat = np.zeros_like(P)
        for m_idx, idx_slice in enumerate(sub_splits):
            best_sub = idx_slice[int(np.argmin(P_F[idx_slice, m_idx]))]
            Y_hat[idx_slice] = P[best_sub]

        if arch_len > 1:
            t_idx = rng.integers(0, arch_len, size=(N, min(3, arch_len)))
            win = np.argmax(cd[t_idx], axis=1)
            A_hat = archive_X[t_idx[np.arange(N), win]]
        else:
            A_hat = archive_X[rng.integers(0, arch_len, size=N)]

        lam = rng.random((N, 1))
        r1, r2, r3 = rng.random((N, d)), rng.random((N, d)), rng.random((N, d))
        V = np.clip(hp.w * V + hp.c1 * r1 * (P - X) + lam * hp.c2 * r2 * (Y_hat - X) + (1.0 - lam) * hp.c3 * r3 * (A_hat - X), -v_max, v_max)
        X = np.clip(X + V, xl, xu)
        F = evaluate_fn(X)

        for m_idx, idx_slice in enumerate(sub_splits):
            improved = F[idx_slice, m_idx] < P_F[idx_slice, m_idx]
            upd_indices = idx_slice[improved]
            P[upd_indices] = X[upd_indices]
            P_F[upd_indices] = F[upd_indices]

        cand_X, cand_F = np.vstack([archive_X, X]), np.vstack([archive_F, F])
        _, u_idx = np.unique(np.round(cand_F, 12), axis=0, return_index=True)
        cand_X, cand_F = cand_X[np.sort(u_idx)], cand_F[np.sort(u_idx)]
        nd_idx = find_non_dominated_indices(cand_F)
        archive_X, archive_F = cand_X[nd_idx], cand_F[nd_idx]
        while len(archive_X) > A_max:
            rem = int(np.argmin(compute_crowding_distance(archive_F)))
            archive_X = np.delete(archive_X, rem, axis=0)
            archive_F = np.delete(archive_F, rem, axis=0)

    return archive_X, archive_F


def run_standard_smpso(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    xl: np.ndarray,
    xu: np.ndarray,
    n_gen: int,
    hp: HyperParameters = HyperParameters(),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Constriction SMPSO Baseline (Nebro et al., 2009)."""
    rng = np.random.default_rng(seed)
    N, A_max, d = hp.swarm_size, hp.archive_size, len(xl)
    delta = (xu - xl) / 2.0

    X = rng.uniform(xl, xu, size=(N, d))
    V = np.zeros((N, d))
    F = evaluate_fn(X)
    P, P_F = X.copy(), F.copy()

    nd_idx = find_non_dominated_indices(P_F)
    archive_X, archive_F = P[nd_idx].copy(), P_F[nd_idx].copy()

    for _ in range(n_gen):
        cd = compute_crowding_distance(archive_F)
        arch_len = len(archive_X)
        if arch_len > 1:
            t_idx = rng.integers(0, arch_len, size=(N, min(2, arch_len)))
            win = np.argmax(cd[t_idx], axis=1)
            Gb = archive_X[t_idx[np.arange(N), win]]
        else:
            Gb = archive_X[rng.integers(0, arch_len, size=N)]

        w = rng.uniform(0.1, 0.5, size=(N, 1))
        c1 = rng.uniform(1.5, 2.5, size=(N, 1))
        c2 = rng.uniform(1.5, 2.5, size=(N, 1))
        phi = np.maximum(4.0001, c1 + c2)
        chi = 2.0 / np.abs(2.0 - phi - np.sqrt(phi ** 2 - 4.0 * phi))
        r1, r2 = rng.random((N, d)), rng.random((N, d))
        V = np.clip(chi * (w * V + c1 * r1 * (P - X) + c2 * r2 * (Gb - X)), -delta, delta)
        X = X + V
        bounce = (X < xl) | (X > xu)
        V[bounce] *= -0.001
        X = np.clip(X, xl, xu)

        turb = (np.arange(N) % 6 == 0)[:, None]
        mask = (rng.random((N, d)) < (1.0 / d)) & turb
        u = rng.random((N, d))
        dq = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (hp.eta + 1.0)) - 1.0, 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (hp.eta + 1.0)))
        X = np.clip(np.where(mask, X + dq * (xu - xl), X), xl, xu)
        F = evaluate_fn(X)

        p_dom = np.all(P_F <= F, axis=1) & np.any(P_F < F, axis=1)
        P[~p_dom], P_F[~p_dom] = X[~p_dom], F[~p_dom]

        cand_X, cand_F = np.vstack([archive_X, X]), np.vstack([archive_F, F])
        _, u_idx = np.unique(np.round(cand_F, 12), axis=0, return_index=True)
        cand_X, cand_F = cand_X[np.sort(u_idx)], cand_F[np.sort(u_idx)]
        nd_idx = find_non_dominated_indices(cand_F)
        archive_X, archive_F = cand_X[nd_idx], cand_F[nd_idx]
        while len(archive_X) > A_max:
            rem = int(np.argmin(compute_crowding_distance(archive_F)))
            archive_X = np.delete(archive_X, rem, axis=0)
            archive_F = np.delete(archive_F, rem, axis=0)

    return archive_X, archive_F


# Built-in ZDT Evaluators & Performance Metrics (IGD, HV)

def get_zdt_problem(name: str, d: int = 30):
    name = name.upper()
    xl, xu = np.zeros(d, dtype=float), np.ones(d, dtype=float)
    n_gen = 300
    t = np.linspace(0.0, 1.0, 500)

    if name == "ZDT1":
        eval_fn = lambda X: np.column_stack([X[:, 0], (1.0 + 9.0 * np.mean(X[:, 1:], axis=1)) * (1.0 - np.sqrt(X[:, 0] / (1.0 + 9.0 * np.mean(X[:, 1:], axis=1))))])
        pf = np.column_stack([t, 1.0 - np.sqrt(t)])
    elif name == "ZDT2":
        eval_fn = lambda X: np.column_stack([X[:, 0], (1.0 + 9.0 * np.mean(X[:, 1:], axis=1)) * (1.0 - np.square(X[:, 0] / (1.0 + 9.0 * np.mean(X[:, 1:], axis=1))))])
        pf = np.column_stack([t, 1.0 - np.square(t)])
    elif name == "ZDT3":
        def eval_fn(X):
            f1 = X[:, 0]
            g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
            return np.column_stack([f1, g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))])
        pf_all = np.column_stack([t, 1.0 - np.sqrt(t) - t * np.sin(10.0 * np.pi * t)])
        pf = pf_all[find_non_dominated_indices(pf_all)]
    elif name == "ZDT4":
        xl[1:], xu[1:], n_gen = -5.0, 5.0, 500
        def eval_fn(X):
            f1 = X[:, 0]
            g = 1.0 + 10.0 * (d - 1) + np.sum(np.square(X[:, 1:]) - 10.0 * np.cos(4.0 * np.pi * X[:, 1:]), axis=1)
            return np.column_stack([f1, g * (1.0 - np.sqrt(f1 / g))])
        pf = np.column_stack([t, 1.0 - np.sqrt(t)])
    elif name == "ZDT6":
        def eval_fn(X):
            f1 = 1.0 - np.exp(-4.0 * X[:, 0]) * np.power(np.sin(6.0 * np.pi * X[:, 0]), 6)
            g = 1.0 + 9.0 * np.power(np.mean(X[:, 1:], axis=1), 0.25)
            return np.column_stack([f1, g * (1.0 - np.square(f1 / g))])
        f1_min = 1.0 - np.exp(-1.0) * (np.sin(1.5 * np.pi) ** 6)
        t6 = np.linspace(f1_min, 1.0, 500)
        pf = np.column_stack([t6, 1.0 - np.square(t6)])
    else:
        raise ValueError(f"Unknown ZDT problem: {name}")

    return eval_fn, xl, xu, pf, n_gen, np.array([2.5, 2.5])


def compute_igd(F: np.ndarray, ref_front: np.ndarray) -> float:
    if HAS_PYMOO:
        return float(PymooIGD(ref_front)(F))
    dists = np.linalg.norm(ref_front[:, None, :] - F[None, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def compute_hv_2d(F: np.ndarray, ref_point: np.ndarray) -> float:
    if HAS_PYMOO:
        return float(PymooHV(ref_point=ref_point)(F))
    valid = F[np.all(F <= ref_point, axis=1)]
    if len(valid) == 0:
        return 0.0
    F_nd = valid[find_non_dominated_indices(valid)]
    F_sorted = F_nd[np.argsort(F_nd[:, 0])]
    hv, prev_f2 = 0.0, ref_point[1]
    for f1, f2 in F_sorted:
        if f2 < prev_f2:
            hv += (ref_point[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MGPSO 2^3 Factorial Ablation & Controls")
    parser.add_argument("--problem", type=str, default="ZDT2", help="Benchmark name (ZDT1, ZDT2, ZDT3, ZDT4, ZDT6)")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")
    args = parser.parse_args()

    eval_fn, xl, xu, pf, n_gen, hv_ref = get_zdt_problem(args.problem)
    print(f"=== Evaluating {args.problem} ({args.runs} runs, n_gen={n_gen}) ===")
    for cfg_name, cfg in ABLATION_CONFIGS.items():
        igds, hvs = [], []
        for r in range(args.runs):
            _, aF = run_mgpso_ablation(eval_fn, xl, xu, n_gen, cfg, seed=1000 + r)
            igds.append(compute_igd(aF, pf))
            hvs.append(compute_hv_2d(aF, hv_ref))
        print(f"  {cfg_name:<10} | IGD: {np.mean(igds):.4f} ± {np.std(igds):.4f} | HV: {np.mean(hvs):.3f} ± {np.std(hvs):.3f}")