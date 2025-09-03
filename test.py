# conformal_pipeline_point_plain.py
# Same pipeline as before but "plain" baseline = point estimate (median under pi_b).
# Conformal methods remain conformalized intervals.
# Requirements: numpy, matplotlib

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import math
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
config = {
    "seed": 1,
    "use_complex_mdp": True,
    "mode": "conformal",
    "weight_method": "mc_kde",  # "none"/"plain_point", "empirical", "regression", "mc_kde", "neural"
    "score": "pinball",
    "alpha": 0.1,
    "mc_sims": 500,
    "diagnostics": True,
    "plot": False,
    "train_frac": 0.6,
    "M": 2400,
    "neural": {"enabled": False},
}

# -----------------------
# MDP and policy utilities (unchanged)
# -----------------------
Trajectory = List[Tuple[int, int, int]]

if config["use_complex_mdp"]:
    N = 10
    TRUE_LAMBDA = 4.0
    H = 8
else:
    N = 5
    TRUE_LAMBDA = 2.0
    H = 5


class ToyMDP:
    def __init__(self, N=N, true_lambda=TRUE_LAMBDA, cost_per_order=0.5, seed=0):
        self.N = N
        self.true_lambda = true_lambda
        self.cost = cost_per_order
        self.rng = np.random.default_rng(seed)

    def step(self, x: int, a: int):
        demand = self.rng.poisson(self.true_lambda)
        next_x = max(0, min(self.N, x + a) - demand)
        sold = max(0, min(self.N, x + a) - next_x)
        reward = sold - (a > 0) * self.cost
        return next_x, int(reward), demand

    def sample_initial(self):
        return int(self.rng.integers(0, self.N + 1))


def oracle_best_action(x: int, N: int):
    return max(0, int(0.6 * N) - x)


def epsilon_greedy_action_probs(x: int, epsilon: float, N: int, A: int):
    best = oracle_best_action(x, N)
    probs = np.full(A, epsilon / A)
    probs[best] += 1.0 - epsilon
    return probs


def sample_action(probs, rng):
    return int(rng.choice(len(probs), p=probs))


def generate_trajectory(
    mdp: ToyMDP, eps_b: float, A: int, rng
) -> Tuple[Trajectory, List[int]]:
    x = mdp.sample_initial()
    traj = []
    demands = []
    for t in range(H):
        probs = epsilon_greedy_action_probs(x, eps_b, mdp.N, A)
        a = sample_action(probs, rng)
        next_x, r, demand = mdp.step(x, a)
        traj.append((x, a, r))
        demands.append(demand)
        x = next_x
    return traj, demands


def traj_return(traj: Trajectory) -> int:
    return sum(r for (_, _, r) in traj)


# -----------------------
# KDE helpers (unchanged)
# -----------------------
def silverman_bandwidth(sample: np.ndarray) -> float:
    n = len(sample)
    if n < 2:
        return 1.0
    sigma = np.std(sample, ddof=1)
    iqr = np.subtract(*np.percentile(sample, [75, 25]))
    sigma_hat = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    if sigma_hat <= 0:
        sigma_hat = max(sigma, 1e-3)
    return 1.06 * sigma_hat * n ** (-1 / 5)


def gaussian_kde_estimate(
    x_values: np.ndarray, eval_point: float, bandwidth: float
) -> float:
    if len(x_values) == 0 or bandwidth <= 0:
        return 1e-12
    diffs = (eval_point - x_values) / bandwidth
    kern = np.exp(-0.5 * diffs * diffs) / math.sqrt(2 * math.pi)
    return float(np.sum(kern) / (len(x_values) * bandwidth))


# -----------------------
# Weight estimators (unchanged)
# -----------------------
def empirical_weight_estimator(
    train_data: List[Trajectory], eps: float, eps_b: float, A: int, N: int
):
    groups = defaultdict(list)
    for traj in train_data:
        x0 = traj[0][0]
        y = traj_return(traj)
        groups[(x0, y)].append(traj)
    weights = {}
    for (x, y), group in groups.items():
        ratios = []
        for traj in group:
            num = 1.0
            den = 1.0
            for xt, at, _ in traj:
                num *= max(epsilon_greedy_action_probs(xt, eps, N, A)[at], 1e-12)
                den *= max(epsilon_greedy_action_probs(xt, eps_b, N, A)[at], 1e-12)
            ratios.append(num / den)
        weights[(x, y)] = float(np.mean(ratios))
    return weights


def regression_weight_estimator(
    train_data: List[Trajectory], eps: float, eps_b: float, A: int, N: int
):
    return empirical_weight_estimator(train_data, eps, eps_b, A, N)


def mc_kde_weight_estimator(
    train_trajs: List[Trajectory],
    train_demands: List[List[int]],
    eps: float,
    eps_b: float,
    A: int,
    N: int,
    rng,
    mc_sims_per_policy=500,
):
    all_demands = [d for seq in train_demands for d in seq]
    lambda_hat = float(np.mean(all_demands)) if len(all_demands) > 0 else 1.0

    xy_candidates = set()
    for traj in train_trajs:
        x0 = traj[0][0]
        y = traj_return(traj)
        xy_candidates.add((x0, y))
    xy_candidates = list(xy_candidates)

    def simulate_returns(x_start: int, epsilon_policy: float, sims: int):
        rets = []
        for _ in range(sims):
            x = x_start
            total_r = 0
            for t in range(H):
                probs = epsilon_greedy_action_probs(x, epsilon_policy, N, A)
                a = sample_action(probs, rng)
                demand = rng.poisson(lambda_hat)
                nxt = max(0, min(N, x + a) - demand)
                sold = max(0, min(N, x + a) - nxt)
                r = sold - (a > 0) * 0.5
                total_r += int(r)
                x = nxt
            rets.append(total_r)
        return np.array(rets, dtype=float)

    p_hat_kde = {}
    for ep in [eps, eps_b]:
        for x, _ in xy_candidates:
            rets = simulate_returns(x, ep, mc_sims_per_policy)
            bw = silverman_bandwidth(rets)
            if bw <= 0:
                bw = 1.0
            p_hat_kde[(ep, x)] = (rets, bw)

    weights = {}
    for x, y in xy_candidates:
        rets_t, bw_t = p_hat_kde[(eps, x)]
        rets_b, bw_b = p_hat_kde[(eps_b, x)]
        p_t = gaussian_kde_estimate(rets_t, float(y), bw_t)
        p_b = gaussian_kde_estimate(rets_b, float(y), bw_b)
        p_t_s = p_t + 1e-12
        p_b_s = p_b + 1e-12
        weights[(x, y)] = float(p_t_s / p_b_s)
    return weights, lambda_hat


# -----------------------
# Score functions and conformal builders (plain now point estimator)
# -----------------------
def compute_state_medians(train_trajs: List[Trajectory]):
    by_state = defaultdict(list)
    for traj in train_trajs:
        x0 = traj[0][0]
        y = traj_return(traj)
        by_state[x0].append(y)
    med = {x: int(np.median(v)) if len(v) > 0 else 0 for x, v in by_state.items()}
    return med


def plain_point_estimate(x_test: int, medians: Dict[int, int]):
    # Plain (non-conformal) baseline: point estimate = median under pi_b
    return medians.get(x_test, 0)


def weighted_conformal_interval(
    calib_trajs: List[Trajectory],
    medians: Dict[int, int],
    weights: Dict[Tuple[int, int], float],
    alpha=0.1,
):
    scored = []
    for traj in calib_trajs:
        x = traj[0][0]
        y = traj_return(traj)
        s = abs(y - medians.get(x, 0))
        w = weights.get((x, y), 1.0)
        scored.append((s, y, w))
    total_w = sum(w for (_, _, w) in scored)
    target_mass = (1.0 - alpha) * total_w
    scored.sort(key=lambda p: p[0])
    cum = 0.0
    threshold = None
    for s, y, w in scored:
        cum += w
        threshold = s
        if cum >= target_mass:
            break
    ys = [y for (s, y, w) in scored if s <= threshold]
    return (min(ys), max(ys)) if ys else (None, None)


# Diagnostics adapted for point baseline
def diagnostics_and_intervals(
    calib_trajs,
    weight_dicts,
    medians,
    alpha=0.1,
    max_print=60,
    do_plot=False,
    rng=None,
    mc_lambda_hat=None,
):
    # build scored list
    scored = []
    for idx, traj in enumerate(calib_trajs):
        x = traj[0][0]
        y = traj_return(traj)
        score = abs(y - medians.get(x, 0))
        weights = {name: wdict.get((x, y), 1.0) for name, wdict in weight_dicts.items()}
        scored.append((idx, x, y, score, weights))

    # compute intervals for each estimator (except plain_point)
    results = {}
    for name, wdict in weight_dicts.items():
        if name == "plain_point":
            continue
        triple = [(s[3], s[2], s[4][name]) for s in scored]
        total_w = sum(w for (_, _, w) in triple)
        target_mass = (1.0 - alpha) * total_w
        triple.sort(key=lambda s: s[0])
        cum = 0.0
        cum_list = []
        threshold = None
        for s_val, y_val, w_val in triple:
            cum += w_val
            cum_list.append((s_val, y_val, w_val, cum))
            threshold = s_val
            if cum >= target_mass:
                break
        ys = [y for (s, y, w) in triple if s <= threshold]
        lo = min(ys) if ys else None
        hi = max(ys) if ys else None
        results[name] = {
            "threshold": threshold,
            "total_w": total_w,
            "target_mass": target_mass,
            "cum_list": cum_list,
            "interval": (lo, hi),
            "all_triple_sorted": triple,
        }

    # print diagnostics
    if config["diagnostics"]:
        print("\nDiagnostics (per-estimator cumulative mass up to cutoff):\n")
        for name, res in results.items():
            print(f"--- Estimator: {name} ---")
            print(
                f"Total weight sum = {res['total_w']:.4f}, target mass (1-alpha) = {res['target_mass']:.4f}"
            )
            print(f"Threshold score (cutoff) = {res['threshold']}")
            print("Top entries up to cutoff (score, y, weight, cumulative_weight):")
            for i, (s_val, y_val, w_val, cum) in enumerate(res["cum_list"][:max_print]):
                print(
                    f"  {i:03d}: score={s_val:6.2f}  y={y_val:3d}  w={w_val:8.4f}  cum={cum:8.4f}"
                )
            print(f"Resulting interval: {res['interval']}\n")

        # side-by-side table including plain point
        print(
            "Side-by-side first calibration samples (idx, x, y, score, "
            + ", ".join(list(weight_dicts.keys()))
            + "):"
        )
        header = "idx  x  y  score  " + "  ".join(
            [f"{k:10s}" for k in weight_dicts.keys()]
        )
        print(header)
        for i, (idx, x, y, score, weights) in enumerate(scored[: min(40, len(scored))]):
            row = f"{idx:3d} {x:3d} {y:3d} {score:7.2f} "
            row += "  ".join(f"{weights.get(k,1.0):10.4f}" for k in weight_dicts.keys())
            print(row)
        print("\n")

    return results


# -----------------------
# Main experiment flow (adapted to point baseline)
# -----------------------
if __name__ == "__main__":
    rng = np.random.default_rng(config["seed"])
    mdp = ToyMDP(N=N, true_lambda=TRUE_LAMBDA, cost_per_order=0.5, seed=config["seed"])
    A = mdp.N + 1
    eps_b = 0.4
    eps = 0.2

    # collect data under behavior policy
    M = config["M"]
    train_frac = config["train_frac"]
    train_trajs = []
    train_demands = []
    calib_trajs = []
    for i in range(M):
        traj, demands = generate_trajectory(mdp, eps_b, A, rng)
        if i < int(train_frac * M):
            train_trajs.append(traj)
            train_demands.append(demands)
        else:
            calib_trajs.append(traj)

    medians = compute_state_medians(train_trajs)

    # compute weights per requested method(s)
    weight_dicts = {}

    # include plain point baseline
    weight_dicts["plain_point"] = {}  # placeholder; handled separately

    # selected weight method plus empirical/regression for comparison
    if config["weight_method"] == "empirical":
        weight_dicts["empirical"] = empirical_weight_estimator(
            train_trajs, eps, eps_b, A, mdp.N
        )
    if config["weight_method"] == "regression":
        weight_dicts["regression"] = regression_weight_estimator(
            train_trajs, eps, eps_b, A, mdp.N
        )
    if config["weight_method"] == "mc_kde":
        mc_weights, lambda_hat = mc_kde_weight_estimator(
            train_trajs,
            train_demands,
            eps,
            eps_b,
            A,
            mdp.N,
            rng,
            mc_sims_per_policy=config["mc_sims"],
        )
        weight_dicts["MC-KDE"] = mc_weights

    # also include empirical/regression for comparison if not present
    if "empirical" not in weight_dicts:
        weight_dicts["empirical"] = empirical_weight_estimator(
            train_trajs, eps, eps_b, A, mdp.N
        )
    if "regression" not in weight_dicts:
        weight_dicts["regression"] = regression_weight_estimator(
            train_trajs, eps, eps_b, A, mdp.N
        )

    # build intervals and diagnostics (plain_point treated separately)
    results = diagnostics_and_intervals(
        calib_trajs,
        weight_dicts,
        medians,
        alpha=config["alpha"],
        max_print=60,
        do_plot=config["plot"],
        rng=rng,
        mc_lambda_hat=(locals().get("lambda_hat", None)),
    )

    # compute plain point predictions for test inputs and evaluate
    def sample_test_trajectories(mdp_obj, eps_target, A, rng_obj, ntests=2000):
        tests = []
        for _ in range(ntests):
            traj, _ = generate_trajectory(mdp_obj, eps_target, A, rng_obj)
            tests.append(traj)
        return tests

    ntests = 2000
    test_trajs = sample_test_trajectories(mdp, eps, A, rng, ntests=ntests)

    # Evaluate conformal intervals and plain point:
    print("Evaluation on test set (under target policy):")
    # Plain point: compute median prediction for each test initial state, measure absolute error
    plain_errors = []
    for traj in test_trajs:
        x0 = traj[0][0]
        y_true = traj_return(traj)
        y_pred = plain_point_estimate(x0, medians)
        plain_errors.append(abs(y_true - y_pred))
    mean_abs_error = float(np.mean(plain_errors))
    print(
        f"Plain point baseline (median under pi_b): mean abs error = {mean_abs_error:.3f}"
    )

    # For each conformal estimator, compute coverage and average interval length
    for name, res in results.items():
        lo, hi = res["interval"]
        if lo is None:
            cov = 0.0
            avg_len = None
        else:
            cov = sum(1 for traj in test_trajs if lo <= traj_return(traj) <= hi) / len(
                test_trajs
            )
            avg_len = hi - lo
        print(
            f"{name:12s} interval={res['interval']}  coverage={cov:.3f}  avg_len={avg_len}"
        )

    # Optional: show how often the plain point is inside each conformal interval
    print(
        "\nFraction of test points where plain point falls inside conformal intervals:"
    )
    for name, res in results.items():
        lo, hi = res["interval"]
        if lo is None:
            frac = 0.0
        else:
            frac = sum(
                1
                for traj in test_trajs
                if lo <= plain_point_estimate(traj[0][0], medians) <= hi
            ) / len(test_trajs)
        print(f"  plain point in {name:12s}: {frac:.3f}")

    # Done
