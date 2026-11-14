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

# 30-run Mean IGD ± Std (Tables 1--3)
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

# 30-run Mean HV ± Std (Tables 4--6, with corrected WFG5 ONLY_PM = 63.366 ± 1.877)
HV_DATA = {
    "ZDT1": [(5.767, 0.050), (5.623, 0.090), (5.733, 0.082), (5.756, 0.047), (5.647, 0.057), (5.664, 0.053), (5.621, 0.050), (5.741, 0.102)],
    "ZDT2": [(5.015, 0.448), (3.569, 0.391), (3.750, 0.000), (4.675, 0.150), (4.063, 0.081), (4.128, 0.112), (3.750, 0.000), (3.621, 0.358)],
    "ZDT3": [(7.026, 0.221), (7.303, 0.091), (6.909, 0.253), (7.318, 0.091), (6.571, 0.225), (7.284, 0.072), (6.479, 0.250), (7.331, 0.116)],
    "ZDT4": [(0.000, 0.000)] * 8,
    "ZDT6": [(5.154, 0.000), (5.036, 0.357), (5.153, 0.000), (5.153, 0.000), (5.154, 0.000), (5.118, 0.193), (5.153, 0.000), (5.153, 0.000)],
    "WFG1": [(44.423, 1.187), (42.901, 1.774), (44.354, 1.173), (42.814, 2.009), (43.313, 1.414), (42.555, 1.897), (44.460, 1.471), (43.153, 1.892)],
    "WFG2": [(90.831, 2.319), (82.856, 5.868), (90.570, 4.237), (89.242, 4.782), (81.685, 4.715), (85.589, 5.097), (81.295, 4.254), (89.462, 4.539)],
    "WFG3": [(63.871, 0.911), (61.747, 0.844), (63.884, 0.895), (62.725, 0.861), (61.830, 1.124), (61.456, 0.862), (62.273, 0.884), (62.809, 0.844)],
    "WFG4": [(69.071, 0.497), (65.793, 7.438), (69.258, 0.576), (68.454, 0.860), (68.798, 0.587), (66.789, 6.342), (68.953, 0.548), (68.333, 0.804)],
    "WFG5": [(51.219, 5.983), (63.236, 1.467), (49.874, 6.287), (54.844, 5.791), (62.653, 2.084), (63.366, 1.877), (62.855, 2.436), (53.672, 6.835)],
    "WFG6": [(60.980, 1.179), (56.931, 0.897), (61.098, 1.429), (58.033, 0.815), (59.047, 1.232), (57.059, 0.839), (58.805, 1.229), (57.761, 1.010)],
    "WFG7": [(61.842, 0.342), (60.530, 0.532), (61.996, 0.381), (61.192, 0.488), (61.510, 0.381), (60.407, 0.561), (61.607, 0.321), (60.937, 0.574)],
    "WFG8": [(55.716, 0.753), (51.025, 1.792), (55.185, 1.136), (52.431, 1.402), (54.436, 1.072), (51.331, 2.850), (54.798, 1.209), (52.332, 1.518)],
    "WFG9": [(66.654, 1.038), (68.304, 1.168), (67.144, 0.781), (67.095, 1.335), (67.385, 1.196), (67.745, 1.366), (67.508, 1.177), (67.672, 1.118)],
}

# Full 2^3 Tchebycheff Factorial on ZDT1--ZDT6 (Table 8)
TCH_ZDT_DATA = {
    "ZDT1": [(0.0033, 0.0003), (0.0072, 0.0032), (0.0021, 0.0002), (0.0019, 0.0002), (0.0110, 0.0036), (0.0092, 0.0034), (0.0080, 0.0030), (0.0174, 0.0464)],
    "ZDT2": [(0.0031, 0.0009), (0.0044, 0.0021), (0.0017, 0.0002), (0.0017, 0.0003), (0.0051, 0.0030), (0.0050, 0.0041), (0.0040, 0.0032), (0.1535, 0.2451)],
    "ZDT3": [(0.0061, 0.0004), (0.0250, 0.0079), (0.0041, 0.0005), (0.0052, 0.0010), (0.0086, 0.0014), (0.0195, 0.0061), (0.0055, 0.0012), (0.0141, 0.0322)],
    "ZDT4": [(124.97, 16.44), (40.35, 10.42), (30.21, 9.27), (106.15, 16.80), (119.38, 16.09), (97.52, 8.29), (22.85, 7.62), (46.69, 7.73)],
    "ZDT6": [(0.00110, 0.00004), (0.00104, 0.00003), (0.00104, 0.00003), (0.00109, 0.00004), (0.00108, 0.00004), (0.00108, 0.00004), (0.00104, 0.00003), (0.43650, 1.30650)],
}


def run_friedman_posthoc(data_dict, label="IGD", higher_is_better=False, exclude_zdt4=True):
    probs = [p for p in data_dict.keys() if not (exclude_zdt4 and p == "ZDT4")]
    K, N_p = 8, len(probs)
    se_rank = np.sqrt(K * (K + 1) / (6.0 * N_p))
    cd_nemenyi = 3.031 * se_rank
    print(f"\n=== {label} Friedman & Post-Hoc Analysis (K={K}, N_p={N_p}, ZDT4 Excluded={exclude_zdt4}) ===")
    print(f"Standard Error (SE_R): {se_rank:.3f} | Nemenyi CD_0.05: {cd_nemenyi:.2f}")

    ranks = []
    for p in probs:
        vals = [v[0] for v in data_dict[p]]
        ranks.append(stats.rankdata([-x for x in vals] if higher_is_better else vals))
    mean_ranks = np.mean(np.array(ranks), axis=0)
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
    print(f"PROPOSED Reference Mean Rank: {rank_dict['PROPOSED']:.2f}")
    print(f"{'Config':<10} | {'Mean Rank':<9} | {'ΔRank':<7} | {'z-stat':<6} | {'p_unadj':<8} | {'Holm α':<8} | {'Holm Sig'} | {'Nemenyi Sig'}")
    print("-" * 88)
    still_sig = True
    for idx, (c, r, diff, z, p) in enumerate(p_list):
        alpha_holm = 0.05 / (7 - idx)
        if p >= alpha_holm:
            still_sig = False
        print(f"{c:<10} | {r:<9.2f} | {diff:+7.2f} | {z:<6.2f} | {p:<8.4f} | {alpha_holm:<8.4f} | {str(still_sig):<8} | {str(diff > cd_nemenyi)}")


def run_factorial_contrasts(data_dict, title="Sum-Scalarized 2^3 Factorial Contrasts (ZDT4 Dropped, WFG1 Flagged)"):
    n = 30
    t_crit = 1.97  # t_{0.975, df=232}
    print(f"\n=== {title} ===")
    print(f"{'Prob':<10} | {'±95% CI':<8} | {'Δ_PM':<9} | {'Δ_TS':<9} | {'Δ_RING':<9} | {'Δ_PM×TS':<9} | {'Δ_TS×RING':<9}")
    print("-" * 78)
    for prob, vals in data_dict.items():
        if prob == "ZDT4":
            continue  
        tag = f"{prob}*" if prob == "WFG1" else prob
        means = {CONFIGS[i]: vals[i][0] for i in range(8)}
        stds  = {CONFIGS[i]: vals[i][1] for i in range(8)}
        se = 0.25 * np.sqrt(sum((stds[c] ** 2) / n for c in CONFIGS))
        ci95 = t_crit * se

        d_pm   = 0.25 * sum(FLAGS[c][0] * means[c] for c in CONFIGS)
        d_ts   = 0.25 * sum(FLAGS[c][1] * means[c] for c in CONFIGS)
        d_ring = 0.25 * sum(FLAGS[c][2] * means[c] for c in CONFIGS)
        d_pm_ts   = 0.25 * sum((FLAGS[c][0] * FLAGS[c][1]) * means[c] for c in CONFIGS)
        d_ts_ring = 0.25 * sum((FLAGS[c][1] * FLAGS[c][2]) * means[c] for c in CONFIGS)
        print(f"{tag:<10} | ±{ci95:<7.4f} | {d_pm:+9.4f} | {d_ts:+9.4f} | {d_ring:+9.4f} | {d_pm_ts:+9.4f} | {d_ts_ring:+9.4f}")


def run_outlier_robust_analysis():
    print("\n=== Outlier-Robust Median & Log10(IGD) Contrasts (ZDT6, WFG4, WFG8) ===")
    for p in ["ZDT6", "WFG4", "WFG8"]:
        vals = IGD_DATA[p]
        means = {CONFIGS[i]: vals[i][0] for i in range(8)}
        stds  = {CONFIGS[i]: vals[i][1] for i in range(8)}
        log10_means, log10_stds, medians = {}, {}, {}
        for c in CONFIGS:
            m, s = means[c], stds[c]
            sig_ln2 = np.log(1.0 + (s / m) ** 2)
            mu_ln = np.log(m) - 0.5 * sig_ln2
            medians[c] = np.exp(mu_ln)
            log10_means[c] = mu_ln / np.log(10.0)
            log10_stds[c] = np.sqrt(sig_ln2) / np.log(10.0)
        ci_log = 1.97 * 0.25 * np.sqrt(sum((log10_stds[c] ** 2) / 30.0 for c in CONFIGS))
        d_pm_log = 0.25 * sum(FLAGS[c][0] * log10_means[c] for c in CONFIGS)
        d_ts_log = 0.25 * sum(FLAGS[c][1] * log10_means[c] for c in CONFIGS)
        d_rg_log = 0.25 * sum(FLAGS[c][2] * log10_means[c] for c in CONFIGS)
        d_tr_log = 0.25 * sum((FLAGS[c][1] * FLAGS[c][2]) * log10_means[c] for c in CONFIGS)
        print(f"{p:<5} | Log10 Contrasts (±{ci_log:.4f}): Δ_PM={d_pm_log:+.4f}, Δ_TS={d_ts_log:+.4f}, Δ_RING={d_rg_log:+.4f}, Δ_TS×RING={d_tr_log:+.4f}")
        print(f"      | Medians: PROPOSED={medians['PROPOSED']:.5f}, BASELINE={medians['BASELINE']:.5f}, ONLY_PM={medians['ONLY_PM']:.5f}")


if __name__ == "__main__":
    run_friedman_posthoc(IGD_DATA, "IGD", higher_is_better=False, exclude_zdt4=True)
    run_friedman_posthoc(HV_DATA,  "HV",  higher_is_better=True,  exclude_zdt4=True)
    run_friedman_posthoc(IGD_DATA, "IGD", higher_is_better=False, exclude_zdt4=False)
    run_friedman_posthoc(HV_DATA,  "HV",  higher_is_better=True,  exclude_zdt4=False)
    run_factorial_contrasts(IGD_DATA)
    run_outlier_robust_analysis()
    run_factorial_contrasts(TCH_ZDT_DATA, title="Tchebycheff-Scalarized 2^3 Factorial Contrasts on ZDT Suite")
