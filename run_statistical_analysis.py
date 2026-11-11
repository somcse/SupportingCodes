import numpy as np
from scipy import stats

CONFIGS = ["PROPOSED", "BASELINE", "NO_PM", "NO_RING", "NO_TS", "ONLY_PM", "ONLY_RING", "ONLY_TS"]

# Orthogonal {-1, +1} coding for (PM, TS, RING) in the 2^3 factorial design
FLAGS = {
    "PROPOSED":  (+1, +1, +1),
    "BASELINE":  (-1, -1, -1),
    "NO_PM":     (-1, +1, +1),
    "NO_RING":   (+1, +1, -1),
    "NO_TS":     (+1, -1, +1),
    "ONLY_PM":   (+1, -1, -1),
    "ONLY_RING": (-1, -1, +1),
    "ONLY_TS":   (-1, +1, -1),
}

IGD_DATA = {
    "ZDT1": [(0.0200, 0.0096), (0.0612, 0.0283), (0.0296, 0.0271), (0.0230, 0.0106), (0.0527, 0.0173), (0.0478, 0.0156), (0.0605, 0.0168), (0.0286, 0.0302)],
    "ZDT2": [(0.0655, 0.0588), (0.6618, 0.1137), (0.6105, 0.0000), (0.1053, 0.0286), (0.3065, 0.0397), (0.2778, 0.0478), (0.6105, 0.0000), (0.6476, 0.1055)],
    "ZDT3": [(0.0490, 0.0251), (0.0318, 0.0155), (0.0659, 0.0381), (0.0254, 0.0189), (0.1207, 0.0465), (0.0312, 0.0116), (0.1442, 0.0560), (0.0220, 0.0167)],
    "ZDT4": [(17.71, 3.02), (16.92, 3.38), (17.49, 3.17), (18.61, 2.27), (17.19, 2.93), (17.32, 4.02), (16.66, 3.03), (17.89, 3.24)],
    "ZDT6": [(0.00114, 0.00007), (0.04726, 0.14050), (0.00121, 0.00009), (0.00113, 0.00009), (0.00115, 0.00006), (0.01404, 0.07078), (0.00121, 0.00010), (0.00123, 0.00008)],
    "WFG1": [(1.441, 0.029), (1.453, 0.033), (1.446, 0.036), (1.466, 0.041), (1.470, 0.041), (1.469, 0.042), (1.435, 0.039), (1.461, 0.043)],
    "WFG2": [(0.289, 0.061), (0.551, 0.142), (0.319, 0.090), (0.324, 0.124), (0.562, 0.079), (0.472, 0.130), (0.607, 0.101), (0.333, 0.108)],
    "WFG3": [(0.284, 0.029), (0.364, 0.029), (0.285, 0.029), (0.320, 0.024), (0.353, 0.031), (0.373, 0.028), (0.338, 0.027), (0.315, 0.024)],
    "WFG4": [(0.238, 0.007), (0.331, 0.232), (0.238, 0.009), (0.251, 0.014), (0.244, 0.010), (0.308, 0.210), (0.241, 0.013), (0.255, 0.014)],
    "WFG5": [(0.590, 0.144), (0.323, 0.026), (0.619, 0.154), (0.506, 0.136), (0.335, 0.045), (0.318, 0.037), (0.327, 0.049), (0.533, 0.160)],
    "WFG6": [(0.398, 0.020), (0.516, 0.020), (0.398, 0.026), (0.462, 0.016), (0.438, 0.025), (0.507, 0.024), (0.445, 0.024), (0.468, 0.022)],
    "WFG7": [(0.401, 0.007), (0.428, 0.016), (0.397, 0.006), (0.419, 0.008), (0.406, 0.007), (0.434, 0.014), (0.403, 0.007), (0.423, 0.010)],
    "WFG8": [(0.512, 0.019), (0.652, 0.059), (0.517, 0.023), (0.580, 0.027), (0.530, 0.021), (0.639, 1.111), (0.522, 0.023), (0.577, 0.033)],
    "WFG9": [(0.271, 0.020), (0.242, 0.021), (0.262, 0.015), (0.265, 0.023), (0.258, 0.022), (0.252, 0.024), (0.252, 0.019), (0.252, 0.019)],
}


def run_friedman_posthoc():
    K, N_p = 8, 14
    se_rank = np.sqrt(K * (K + 1) / (6.0 * N_p))
    cd_nemenyi = 3.031 * se_rank
    print(f"=== Friedman Post-Hoc Analysis (K={K}, N_p={N_p}) ===")
    print(f"Standard Error of Rank Difference (SE_R): {se_rank:.4f}")
    print(f"Nemenyi Critical Difference (CD_0.05):    {cd_nemenyi:.2f}\n")

    ranks_matrix = np.array([stats.rankdata([v[0] for v in vals]) for vals in IGD_DATA.values()])
    mean_ranks = np.mean(ranks_matrix, axis=0)
    rank_dict = {CONFIGS[i]: mean_ranks[i] for i in range(K)}

    p_list = []
    for c, r in rank_dict.items():
        if c == "PROPOSED":
            continue
        diff = r - rank_dict["PROPOSED"]
        z = diff / se_rank
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        p_list.append((c, r, diff, z, p))

    p_list.sort(key=lambda x: x[4])
    print(f"{'Config':<10} | {'Mean Rank':<9} | {'ΔRank':<7} | {'z-stat':<6} | {'p_unadj':<8} | {'Holm α':<8} | {'Holm Sig'} | {'Nemenyi Sig'}")
    print("-" * 86)
    for idx, (c, r, diff, z, p) in enumerate(p_list):
        alpha_holm = 0.05 / (7 - idx)
        print(f"{c:<10} | {r:<9.2f} | +{diff:<6.2f} | {z:<6.2f} | {p:<8.4f} | {alpha_holm:<8.4f} | {str(p < alpha_holm):<8} | {str(diff > cd_nemenyi)}")


def run_factorial_contrasts():
    n = 30
    t_crit = 1.97  # t_{0.975, df=232}
    print("\n=== 2^3 Factorial Main Effects & Two-Way Interactions (Eq. 9, ± 95% CI) ===")
    print(f"{'Prob':<6} | {'±95% CI':<8} | {'Δ_PM':<9} | {'Δ_TS':<9} | {'Δ_RING':<9} | {'Δ_PM×TS':<9} | {'Δ_TS×RING':<9}")
    print("-" * 74)
    for prob, vals in IGD_DATA.items():
        means = {CONFIGS[i]: vals[i][0] for i in range(8)}
        stds  = {CONFIGS[i]: vals[i][1] for i in range(8)}
        se = 0.25 * np.sqrt(sum((stds[c] ** 2) / n for c in CONFIGS))
        ci95 = t_crit * se

        d_pm   = 0.25 * sum(FLAGS[c][0] * means[c] for c in CONFIGS)
        d_ts   = 0.25 * sum(FLAGS[c][1] * means[c] for c in CONFIGS)
        d_ring = 0.25 * sum(FLAGS[c][2] * means[c] for c in CONFIGS)
        d_pm_ts   = 0.25 * sum((FLAGS[c][0] * FLAGS[c][1]) * means[c] for c in CONFIGS)
        d_ts_ring = 0.25 * sum((FLAGS[c][1] * FLAGS[c][2]) * means[c] for c in CONFIGS)
        print(f"{prob:<6} | ±{ci95:<7.4f} | {d_pm:+9.4f} | {d_ts:+9.4f} | {d_ring:+9.4f} | {d_pm_ts:+9.4f} | {d_ts_ring:+9.4f}")


if __name__ == "__main__":
    run_friedman_posthoc()
    run_factorial_contrasts()