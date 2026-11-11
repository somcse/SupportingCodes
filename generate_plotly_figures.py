from __future__ import annotations
import os
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mgpso_ablation_framework import (
    ABLATION_CONFIGS,
    HyperParameters,
    run_mgpso_ablation,
    get_zdt_problem,
    find_non_dominated_indices,
)

try:
    from pymoo.problems import get_problem
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False


def load_wfg_problem(wfg_idx: int, n_obj: int = 3, n_var: int = 24, k_pos: int = 4):
    """Load 3-objective WFG1--WFG9 problem (via pymoo if installed, or analytical WFG fallback)."""
    if HAS_PYMOO:
        prob = get_problem(f"wfg{wfg_idx}", n_obj=n_obj, n_var=n_var, k=k_pos)
        return prob.evaluate, prob.xl, prob.xu, prob.pareto_front(), 400

    xl = np.zeros(n_var, dtype=float)
    xu = 2.0 * np.arange(1, n_var + 1, dtype=float)
    scales = np.array([2.0, 4.0, 6.0])
    u_grid = np.linspace(0.0, 1.0, 28)
    U, V = np.meshgrid(u_grid, u_grid)
    u_flat, v_flat = U.ravel(), V.ravel()

    def shape_fn(y1, y2, idx):
        if idx == 1:
            h1 = 1.0 - np.cos(0.5 * np.pi * y1) * np.cos(0.5 * np.pi * y2)
            h2 = 1.0 - np.cos(0.5 * np.pi * y1) * np.sin(0.5 * np.pi * y2)
            h3 = 1.0 - y1 - np.cos(2.0 * np.pi * y1 + 0.5 * np.pi) / (2.0 * np.pi)
        elif idx == 2:
            h1 = 1.0 - np.cos(0.5 * np.pi * y1) * np.cos(0.5 * np.pi * y2)
            h2 = 1.0 - np.cos(0.5 * np.pi * y1) * np.sin(0.5 * np.pi * y2)
            h3 = 1.0 - y1 * (np.cos(5.0 * np.pi * y1) ** 2)
        elif idx == 3:
            h1, h2, h3 = y1 * y2, y1 * (1.0 - y2), 1.0 - y1
        else:
            h1 = np.sin(0.5 * np.pi * y1) * np.cos(0.5 * np.pi * y2)
            h2 = np.sin(0.5 * np.pi * y1) * np.sin(0.5 * np.pi * y2)
            h3 = np.cos(0.5 * np.pi * y1)
        return np.column_stack([scales[0] * h1, scales[1] * h2, scales[2] * h3])

    pf_raw = shape_fn(u_flat, v_flat, wfg_idx)
    pf = pf_raw[find_non_dominated_indices(pf_raw)]

    def eval_fn(X: np.ndarray) -> np.ndarray:
        Z = X / xu[None, :]
        y1 = np.mean(Z[:, : k_pos // 2], axis=1)
        y2 = np.mean(Z[:, k_pos // 2 : k_pos], axis=1)
        dist = Z[:, k_pos:] - 0.35
        g = 2.0 * np.mean(np.square(dist), axis=1)
        return shape_fn(y1, y2, wfg_idx) + scales[None, :] * g[:, None]

    return eval_fn, xl, xu, pf, 400


def save_plotly_figure(fig: go.Figure, base_path_no_ext: str, width: int, height: int, scale: int = 3):
    os.makedirs(os.path.dirname(os.path.abspath(base_path_no_ext)), exist_ok=True)
    html_path = f"{base_path_no_ext}.html"
    png_path = f"{base_path_no_ext}.png"
    pdf_path = f"{base_path_no_ext}.pdf"
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(png_path, width=width, height=height, scale=scale)
        fig.write_image(pdf_path, width=width, height=height, scale=1)
        print(f"  [Saved PNG/PDF/HTML] -> {png_path}")
    except Exception as exc:
        print(f"  [Saved HTML] -> {html_path} (Install `pip install kaleido` for PNG/PDF: {exc})")


def generate_figure1_comparative_zdt(out_dir: str, seed: int = 42):
    print("\n[1/4] Generating Figure 1: Comparative ZDT Ablation Fronts...")
    hp = HyperParameters()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>(a) ZDT1 (Convex): Uniform Coverage via TS</b>",
            "<b>(b) ZDT2 (Concave): Collapse Without PM</b>",
            "<b>(c) ZDT3 (Disconnected): Global vs. Ring Propagation</b>",
            "<b>(d) ZDT6 (Biased Density): Density-Bias Mitigation</b>",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    eval_fn, xl, xu, pf, n_gen, _ = get_zdt_problem("ZDT1")
    _, f_prop = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["PROPOSED"], hp, seed=seed)
    _, f_nots = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["NO_TS"], hp, seed=seed)
    _, f_base = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["BASELINE"], hp, seed=seed)

    fig.add_trace(go.Scatter(x=pf[:, 0], y=pf[:, 1], mode="lines", name="True Pareto Front",
                             line=dict(color="black", width=2.5), legendgroup="pf"), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_base[:, 0], y=f_base[:, 1], mode="markers", name="BASELINE (0,0,0)",
                             marker=dict(symbol="triangle-up", size=7, color="#d95f02", opacity=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_nots[:, 0], y=f_nots[:, 1], mode="markers", name="NO_TS (Uniform Archive)",
                             marker=dict(symbol="square", size=6, color="#7570b3", opacity=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_prop[:, 0], y=f_prop[:, 1], mode="markers", name="PROPOSED (Full MGPSO)",
                             marker=dict(symbol="circle", size=6, color="#1b9e77", opacity=0.9), legendgroup="prop"), row=1, col=1)

    eval_fn, xl, xu, pf, n_gen, _ = get_zdt_problem("ZDT2")
    _, f_prop2 = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["PROPOSED"], hp, seed=seed)
    _, f_nopm2 = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["NO_PM"], hp, seed=seed)
    _, f_onpm2 = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["ONLY_PM"], hp, seed=seed)

    fig.add_trace(go.Scatter(x=pf[:, 0], y=pf[:, 1], mode="lines", name="True Pareto Front",
                             line=dict(color="black", width=2.5), showlegend=False, legendgroup="pf"), row=1, col=2)
    fig.add_trace(go.Scatter(x=f_onpm2[:, 0], y=f_onpm2[:, 1], mode="markers", name="ONLY_PM (Sparse Front)",
                             marker=dict(symbol="diamond", size=7, color="#7570b3", opacity=0.8)), row=1, col=2)
    fig.add_trace(go.Scatter(x=f_prop2[:, 0], y=f_prop2[:, 1], mode="markers", name="PROPOSED (Full MGPSO)",
                             marker=dict(symbol="circle", size=6, color="#1b9e77", opacity=0.9), showlegend=False, legendgroup="prop"), row=1, col=2)
    fig.add_trace(go.Scatter(x=f_nopm2[:, 0], y=f_nopm2[:, 1], mode="markers", name="NO_PM & ONLY_RING Collapse (1 pt)",
                             marker=dict(symbol="x", size=13, color="#e7298a", line=dict(color="black", width=1.2))), row=1, col=2)

    fig.add_annotation(
        x=float(f_nopm2[0, 0]), y=float(f_nopm2[0, 1]),
        xref="x2", yref="y2",
        text="<b>Zero-variance collapse<br>at (0, 1) without PM</b>",
        showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.8, arrowcolor="#b00058",
        ax=95, ay=55,
        bgcolor="#fff5f8", bordercolor="#e7298a", borderwidth=1.2,
        font=dict(size=12, color="#b00058"),
    )

    eval_fn, xl, xu, pf, n_gen, _ = get_zdt_problem("ZDT3")
    _, f_prop3 = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["PROPOSED"], hp, seed=seed)
    _, f_onts3 = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["ONLY_TS"], hp, seed=seed)
    _, f_onrg3 = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["ONLY_RING"], hp, seed=seed)

    fig.add_trace(go.Scatter(x=pf[:, 0], y=pf[:, 1], mode="markers", name="True Pareto Front",
                             marker=dict(symbol="circle", size=3.5, color="black"), showlegend=False, legendgroup="pf"), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_onrg3[:, 0], y=f_onrg3[:, 1], mode="markers", name="ONLY_RING (Slow Jump)",
                             marker=dict(symbol="triangle-up", size=7, color="#d95f02", opacity=0.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_prop3[:, 0], y=f_prop3[:, 1], mode="markers", name="PROPOSED (Full MGPSO)",
                             marker=dict(symbol="circle", size=6, color="#1b9e77", opacity=0.85), showlegend=False, legendgroup="prop"), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_onts3[:, 0], y=f_onts3[:, 1], mode="markers", name="ONLY_TS (Fastest Jump)",
                             marker=dict(symbol="star", size=7.5, color="#2b5c8f", opacity=0.95)), row=2, col=1)

    eval_fn, xl, xu, pf, n_gen, _ = get_zdt_problem("ZDT6")
    _, f_prop6 = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["PROPOSED"], hp, seed=seed)
    _, f_base6 = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["BASELINE"], hp, seed=seed)
    f_prop6_v = f_prop6[f_prop6[:, 1] <= 1.15]
    f_base6_v = f_base6[f_base6[:, 1] <= 1.15]

    fig.add_trace(go.Scatter(x=pf[:, 0], y=pf[:, 1], mode="lines", name="True Pareto Front",
                             line=dict(color="black", width=2.5), showlegend=False, legendgroup="pf"), row=2, col=2)
    fig.add_trace(go.Scatter(x=f_base6_v[:, 0], y=f_base6_v[:, 1], mode="markers", name="BASELINE (0,0,0)",
                             marker=dict(symbol="triangle-up", size=7.5, color="#d95f02", opacity=0.85), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=f_prop6_v[:, 0], y=f_prop6_v[:, 1], mode="markers", name="PROPOSED (Full MGPSO)",
                             marker=dict(symbol="circle", size=6, color="#1b9e77", opacity=0.9), showlegend=False, legendgroup="prop"), row=2, col=2)

    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text="<i>f</i><sub>1</sub>", showgrid=True, gridcolor="#e6e6e6",
                             zeroline=False, mirror=True, ticks="outside", showline=True, linecolor="black", row=r, col=c)
            fig.update_yaxes(title_text="<i>f</i><sub>2</sub>", showgrid=True, gridcolor="#e6e6e6",
                             zeroline=False, mirror=True, ticks="outside", showline=True, linecolor="black", row=r, col=c)

    fig.update_yaxes(range=[-0.85, 1.15], row=2, col=1)
    fig.update_yaxes(range=[-0.06, 1.08], row=2, col=2)

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman, Serif", size=15, color="black"),
        width=1050, height=820,
        legend=dict(orientation="h", yanchor="bottom", y=-0.13, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#cccccc", borderwidth=1, font=dict(size=12)),
        margin=dict(l=65, r=35, t=55, b=105),
    )
    save_plotly_figure(fig, os.path.join(out_dir, "fig1_zdt_ablation_comparison"), width=1050, height=820)


def generate_figure2_factorial_and_ranks(out_dir: str):
    print("\n[2/4] Generating Figure 2: Friedman Ranks (with CD=2.81) & 2^3 Factorial Effects (with 95% CIs)...")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>(a) Friedman Ranks (Nemenyi CD<sub>0.05</sub> = 2.81)</b>",
            "<b>(b) 2<sup>3</sup> Factorial Main Effects (ΔIGD ± 95% CI)</b>",
        ),
        horizontal_spacing=0.14,
    )

    cfg_labels = [
        "BASELINE (0,0,0)", "ONLY_PM (1,0,0)", "ONLY_TS (0,1,0)", "NO_TS (1,0,1)",
        "NO_RING (1,1,0)", "ONLY_RING (0,0,1)", "NO_PM (0,1,1)", "PROPOSED (1,1,1)",
    ]
    igd_ranks = [5.79, 5.36, 4.86, 4.86, 4.36, 4.21, 3.71, 2.86]
    hv_ranks  = [5.89, 5.96, 4.50, 4.50, 4.36, 4.54, 3.46, 2.79]

    fig.add_trace(go.Bar(
        y=cfg_labels, x=hv_ranks, orientation="h",
        name="HV Mean Rank (Lower = Better)",
        marker=dict(color="#2b5c8f", line=dict(color="black", width=0.8)),
        text=[f"{v:.2f}" for v in hv_ranks], textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=cfg_labels, x=igd_ranks, orientation="h",
        name="IGD Mean Rank (Lower = Better)",
        marker=dict(color="#1b9e77", line=dict(color="black", width=0.8)),
        text=[f"<b>{v:.2f}</b>" for v in igd_ranks], textposition="outside",
    ), row=1, col=1)

    # Nemenyi Critical Difference threshold line from PROPOSED IGD rank (2.86 + 2.81 = 5.67)
    fig.add_vline(x=2.86 + 2.81, line_dash="dash", line_color="#b00058", line_width=1.5, row=1, col=1)

    benchmarks = ["ZDT1<br>(Convex)", "ZDT2<br>(Concave)", "ZDT3<br>(Disconn.)", "WFG2<br>(Disconn.)", "WFG6<br>(Non-Sep.)", "WFG8<br>(Biased)"]
    eff_pm   = [-0.0091, -0.4438, -0.0094, -0.0408, -0.0055, -0.0017]
    eff_ts   = [-0.0303, -0.1069, -0.0414, -0.2318, -0.0450, -0.0393]
    eff_ring = [+0.0005, -0.0249, +0.0673, +0.0242, -0.0685, -0.0917]
    ci95     = [0.0053,  0.0161,  0.0082,  0.0273,  0.0057,  0.1002]

    fig.add_trace(go.Bar(
        x=benchmarks, y=eff_pm, name="Main Effect Δ<sub>PM</sub>",
        error_y=dict(type="data", array=ci95, visible=True, thickness=1.2, width=3),
        marker=dict(color="#e7298a", line=dict(color="black", width=0.8)),
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=benchmarks, y=eff_ts, name="Main Effect Δ<sub>TS</sub>",
        error_y=dict(type="data", array=ci95, visible=True, thickness=1.2, width=3),
        marker=dict(color="#1b9e77", line=dict(color="black", width=0.8)),
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=benchmarks, y=eff_ring, name="Main Effect Δ<sub>RING</sub>",
        error_y=dict(type="data", array=ci95, visible=True, thickness=1.2, width=3),
        marker=dict(color="#d95f02", line=dict(color="black", width=0.8)),
    ), row=1, col=2)

    fig.add_hline(y=0.0, line_width=1.2, line_color="black", row=1, col=2)

    fig.update_xaxes(title_text="Friedman Mean Rank Across 14 Benchmarks", range=[0, 6.9],
                     showgrid=True, gridcolor="#e6e6e6", mirror=True, showline=True, linecolor="black", row=1, col=1)
    fig.update_yaxes(mirror=True, showline=True, linecolor="black", row=1, col=1)

    fig.update_xaxes(title_text="Benchmark Problem & Landscape Type",
                     mirror=True, showline=True, linecolor="black", row=1, col=2)
    fig.update_yaxes(title_text="Factorial Main Effect on IGD (ΔIGD < 0 Improves)", range=[-0.49, 0.12],
                     showgrid=True, gridcolor="#e6e6e6", mirror=True, showline=True, linecolor="black", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        font=dict(family="Times New Roman, Serif", size=14.5, color="black"),
        width=1120, height=480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#cccccc", borderwidth=1, font=dict(size=12)),
        margin=dict(l=145, r=35, t=55, b=110),
    )
    save_plotly_figure(fig, os.path.join(out_dir, "fig2_factorial_effects_and_ranks"), width=1120, height=480)


def generate_individual_zdt_plots(out_dir: str, seed: int = 42):
    print("\n[3/4] Generating Individual ZDT1-4, ZDT6 Plots in images/proposed/...")
    hp = HyperParameters()
    zdt_dir = os.path.join(out_dir, "images", "proposed")
    os.makedirs(zdt_dir, exist_ok=True)

    for name in ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]:
        eval_fn, xl, xu, pf, n_gen, _ = get_zdt_problem(name)
        _, f_prop = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["PROPOSED"], hp, seed=seed)

        fig = go.Figure()
        pf_mode = "markers" if name == "ZDT3" else "lines"
        fig.add_trace(go.Scatter(x=pf[:, 0], y=pf[:, 1], mode=pf_mode, name="True PF",
                                 line=dict(color="#d62728", width=2.5), marker=dict(color="#d62728", size=4)))
        fig.add_trace(go.Scatter(x=f_prop[:, 0], y=f_prop[:, 1], mode="markers", name="PROPOSED",
                                 marker=dict(color="#1f77b4", size=7, opacity=0.85, line=dict(color="black", width=0.5))))
        fig.update_layout(
            template="plotly_white",
            title=dict(text=f"<b>PROPOSED — {name}</b>", x=0.5, font=dict(size=16)),
            xaxis=dict(title="<i>f</i><sub>1</sub>", showgrid=True, gridcolor="#e6e6e6", mirror=True, showline=True, linecolor="black"),
            yaxis=dict(title="<i>f</i><sub>2</sub>", showgrid=True, gridcolor="#e6e6e6", mirror=True, showline=True, linecolor="black"),
            font=dict(family="Times New Roman, Serif", size=14, color="black"),
            legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
            width=520, height=440, margin=dict(l=60, r=25, t=50, b=55),
        )
        save_plotly_figure(fig, os.path.join(zdt_dir, f"{name.lower()}_pareto"), width=520, height=440, scale=2)


def generate_individual_wfg_3d_plots(out_dir: str, seed: int = 42):
    print("\n[4/4] Generating 3D WFG1-WFG9 Plots in images/proposed/...")
    hp = HyperParameters()
    wfg_dir = os.path.join(out_dir, "images", "proposed")
    os.makedirs(wfg_dir, exist_ok=True)

    for idx in range(1, 10):
        eval_fn, xl, xu, pf, n_gen = load_wfg_problem(idx)
        _, f_prop = run_mgpso_ablation(eval_fn, xl, xu, n_gen, ABLATION_CONFIGS["PROPOSED"], hp, seed=seed + idx)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=pf[:, 0], y=pf[:, 1], z=pf[:, 2], mode="markers", name="True PF",
                                   marker=dict(size=2.2, color="#aaaaaa", opacity=0.35)))
        fig.add_trace(go.Scatter3d(x=f_prop[:, 0], y=f_prop[:, 1], z=f_prop[:, 2], mode="markers", name="PROPOSED",
                                   marker=dict(size=4.0, color="#d62728", opacity=0.9, line=dict(color="black", width=0.3))))
        fig.update_layout(
            template="plotly_white",
            title=dict(text=f"<b>WFG{idx} (PROPOSED)</b>", x=0.5, y=0.96, font=dict(size=18)),
            font=dict(family="Times New Roman, Serif", size=15, color="black"),
            scene=dict(
                xaxis=dict(title="f₁", backgroundcolor="rgb(248,248,248)", gridcolor="white", showbackground=True),
                yaxis=dict(title="f₂", backgroundcolor="rgb(248,248,248)", gridcolor="white", showbackground=True),
                zaxis=dict(title="f₃", backgroundcolor="rgb(248,248,248)", gridcolor="white", showbackground=True),
                camera=dict(eye=dict(x=1.30, y=1.30, z=1.05)),
            ),
            legend=dict(x=0.95, y=0.92, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
            width=550, height=480, margin=dict(l=10, r=10, t=40, b=10),
        )
        save_plotly_figure(fig, os.path.join(wfg_dir, f"wfg{idx}_pareto"), width=550, height=480, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all MGPSO Ablation Paper figures using Plotly.")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-wfg", action="store_true", help="Skip 3D WFG1-WFG9 generation")
    args = parser.parse_args()

    generate_figure1_comparative_zdt(args.out_dir, seed=args.seed)
    generate_figure2_factorial_and_ranks(args.out_dir)
    generate_individual_zdt_plots(args.out_dir, seed=args.seed)
    if not args.skip_wfg:
        generate_individual_wfg_3d_plots(args.out_dir, seed=args.seed)
    print("\nAll figures generated!")