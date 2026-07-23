"""Generate the complete paper figure and table suite."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import LogFormatterSciNotation, ScalarFormatter
import numpy as np

from channel_constant import generate_irs_channels, init_geometry_and_channels
from channels import (
    compute_active_irs_noise_power,
    compute_active_irs_output_power,
    compute_effective_a,
    compute_effective_h,
    compute_forwarded_irs_sensing_noise_power,
    compute_safe_uniform_active_gain,
    irs_beam_align,
)
from config import (
    A_MAX,
    Mr,
    Mt,
    N_irs_list,
    P,
    P_RIS,
    P_RIS_dBm,
    SEED,
    SEED_CHANNEL,
    SEED_IRS_CU,
    SIGMA2_RIS,
    SIGMA2_RIS_dBm,
    T,
    alpha0,
    d0,
    gamma_0_dB_max,
    gamma_0_dB_min,
    K0,
    Kc,
    pos_bs,
    pos_irs,
    pos_rx,
    pos_target,
    sigma2_c,
    sigma2_c_dBm,
    sigma2_s,
    sigma2_s_dBm,
)
from crb import compute_crb_irs
from rate import compute_rate_irs
from sca_solver import solve_p4_sca


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "paper_figures"
CACHE_PATH = OUTPUT_DIR / "metric_cache.json"

COLORS = {
    "los": "#111111",
    "passive": "#1f77b4",
    "active": "#d62728",
    "noise": "#9467bd",
}
N_COLORS = {
    8: "#7f7f7f",
    16: "#1f77b4",
    24: "#17becf",
    32: "#2ca02c",
    40: "#bcbd22",
    64: "#ff7f0e",
    96: "#9467bd",
    128: "#d62728",
}
GAMMA_STYLES = {0.0: "-", 10.0: "--", 16.0: ":"}


class MetricCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.path.write_text(
            json.dumps(self.data, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class Experiment:
    def __init__(self):
        np.random.seed(SEED)
        self.geo, self.ch = init_geometry_and_channels()
        self.irs_channels = {}
        self.cache = MetricCache(CACHE_PATH)

    def irs(self, n_irs):
        if n_irs not in self.irs_channels:
            self.irs_channels[n_irs] = generate_irs_channels(
                n_irs, pos_irs, self.geo
            )
        return self.irs_channels[n_irs]

    def solve(
        self,
        mode,
        gamma_db,
        n_irs=0,
        bs_power=P,
        amax=A_MAX,
        ris_power=P_RIS,
        sigma2_irs=SIGMA2_RIS,
    ):
        key = "|".join([
            mode,
            f"N={n_irs}",
            f"g={gamma_db:.8g}",
            f"P={bs_power:.8e}",
            f"A={amax:.8e}",
            f"Pr={ris_power:.8e}",
            f"Si={sigma2_irs:.8e}",
        ])
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        gamma = 10 ** (gamma_db / 10)
        start = time.perf_counter()
        if mode == "los":
            a_eff = self.ch["a_dir"]
            h_eff = self.ch["h"]
            v = None
            irs_noise = 0.0
        else:
            irs_ch = self.irs(n_irs)
            v = irs_beam_align(irs_ch["h_r"], irs_ch["G"])
            if mode in {"active", "active_ideal"}:
                effective_sigma = 0.0 if mode == "active_ideal" else sigma2_irs
                gain = compute_safe_uniform_active_gain(
                    irs_ch["G"], bs_power, ris_power,
                    effective_sigma, amax,
                )
                v *= gain
                irs_noise = compute_active_irs_noise_power(
                    irs_ch["h_rc"], v, effective_sigma
                )
            else:
                effective_sigma = 0.0
                irs_noise = 0.0
            a_eff = compute_effective_a(
                self.ch["a_dir"], irs_ch["G"], irs_ch["h_r"], v,
                direct_blocked=True,
            )
            h_eff = compute_effective_h(
                self.ch["h"], irs_ch["G"], irs_ch["h_rc"], v
            )

        Rc, Rs, info = solve_p4_sca(
            gamma, h_eff, a_eff, sigma2_c, sigma2_s,
            bs_power, Mt, Mr, self.ch["b"], self.ch["b_dot"],
            self.ch["alpha_sq"], extra_noise_power=irs_noise,
        )
        elapsed = time.perf_counter() - start
        if Rc is None:
            result = {"feasible": False, "status": info["status"]}
            self.cache.put(key, result)
            return result

        rate, sinr = compute_rate_irs(
            Rc, Rs, h_eff, sigma2_c, irs_noise
        )
        crb = compute_crb_irs(
            Rc, Rs, a_eff, self.ch["b"], self.ch["b_dot"],
            self.ch["alpha_sq"], sigma2_s, T,
        )
        if v is None:
            gain = irs_output = forwarded_noise = 0.0
        else:
            gain = float(np.max(np.abs(v)))
            if mode in {"active", "active_ideal"}:
                irs_output = compute_active_irs_output_power(
                    irs_ch["G"], Rc, Rs, v, effective_sigma
                )
                forwarded_noise = (
                    compute_forwarded_irs_sensing_noise_power(
                        irs_ch["h_r"], v, self.ch["b"],
                        self.ch["alpha_sq"], effective_sigma,
                    )
                )
            else:
                irs_output = forwarded_noise = 0.0

        result = {
            "feasible": True,
            "crb": float(crb),
            "rate": float(rate),
            "sinr": float(sinr),
            "runtime_s": float(elapsed),
            "gain": gain,
            "irs_output_power": float(irs_output),
            "irs_noise_power": float(irs_noise),
            "forwarded_sensing_noise": float(forwarded_noise),
            "status": info["status"],
        }
        self.cache.put(key, result)
        return result


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
    })


def format_crb_axis(ax, log=False):
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set_ylabel(r"CRB (rad$^2$)")


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


def valid_series(results, x_name):
    valid = [r for r in results if r["metric"]["feasible"]]
    return (
        np.array([r[x_name] for r in valid]),
        np.array([r["metric"]["crb"] for r in valid]),
    )


def sweep_gamma(exp, mode, n_irs, gamma_grid, bs_power=P, **kwargs):
    return [
        {
            "gamma_db": float(gamma_db),
            "rate": exp.solve(
                mode, gamma_db, n_irs, bs_power=bs_power, **kwargs
            ).get("rate", np.nan),
            "metric": exp.solve(
                mode, gamma_db, n_irs, bs_power=bs_power, **kwargs
            ),
        }
        for gamma_db in gamma_grid
    ]


def figure_01_system_model():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.4, 5.2)
    ax.axis("off")

    def node(x, y, label, color, shape="box"):
        if shape == "circle":
            patch = Circle((x, y), 0.38, facecolor=color, edgecolor="black")
        elif shape == "diamond":
            patch = Polygon(
                [(x, y + 0.48), (x + 0.48, y), (x, y - 0.48),
                 (x - 0.48, y)],
                facecolor=color, edgecolor="black",
            )
        else:
            patch = FancyBboxPatch(
                (x - 0.55, y - 0.35), 1.1, 0.7,
                boxstyle="round,pad=0.08", facecolor=color,
                edgecolor="black",
            )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", weight="bold")

    node(0.7, 2.8, "BS", "#9ecae1")
    node(3.7, 4.2, "Active IRS", "#fdae6b")
    node(6.6, 3.5, "Target", "#fdd0a2", "diamond")
    node(9.3, 2.4, "Sensing RX", "#bcbddc")
    node(6.4, 0.8, "CU", "#a1d99b", "circle")

    def arrow(start, end, color, style, label, offset=(0, 0)):
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", color=color, lw=2.2, ls=style),
        )
        mid = ((start[0] + end[0]) / 2 + offset[0],
               (start[1] + end[1]) / 2 + offset[1])
        ax.text(*mid, label, color=color, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    arrow((1.2, 3.0), (3.15, 4.0), "#d95f02", "-", "BS→IRS")
    arrow((4.25, 4.1), (6.1, 3.6), "#d95f02", "-", "IRS→Target")
    arrow((7.1, 3.35), (8.75, 2.55), "#d95f02", "-", "Target→RX")
    arrow((1.2, 2.55), (6.1, 3.35), "#636363", "--",
          "direct sensing path", (0, -0.25))
    arrow((1.15, 2.5), (6.05, 1.0), "#1b9e77", "-", "BS→CU")
    arrow((4.0, 3.85), (6.2, 1.15), "#1b9e77", "--", "IRS→CU")

    noise = Rectangle(
        (3.25, 2.8), 0.9, 0.55, facecolor="#f7f7f7",
        edgecolor="#756bb1", hatch="///",
    )
    ax.add_patch(noise)
    ax.text(3.7, 3.075, r"$\mathbf{z}_I$", ha="center", va="center")
    arrow((3.7, 3.35), (3.7, 3.78), "#756bb1", ":", "amplifier noise",
          (1.0, 0.0))
    ax.text(
        5.0, -0.05,
        "Single-pass sensing: the echo does not return through the IRS",
        ha="center", fontsize=10, style="italic",
    )
    save_figure(fig, "fig01_system_model")


def convergence_history(exp, mode, n_irs=40, gamma_db=10.0):
    irs_ch = exp.irs(n_irs)
    v = irs_beam_align(irs_ch["h_r"], irs_ch["G"])
    if mode == "active":
        v *= compute_safe_uniform_active_gain(
            irs_ch["G"], P, P_RIS, SIGMA2_RIS, A_MAX
        )
        irs_noise = compute_active_irs_noise_power(
            irs_ch["h_rc"], v, SIGMA2_RIS
        )
    else:
        irs_noise = 0.0
    a_eff = compute_effective_a(
        exp.ch["a_dir"], irs_ch["G"], irs_ch["h_r"], v,
        direct_blocked=True,
    )
    h_eff = compute_effective_h(
        exp.ch["h"], irs_ch["G"], irs_ch["h_rc"], v
    )
    _, _, info = solve_p4_sca(
        10 ** (gamma_db / 10), h_eff, a_eff,
        sigma2_c, sigma2_s, P, Mt, Mr,
        exp.ch["b"], exp.ch["b_dot"], exp.ch["alpha_sq"],
        max_iter=30, tol=1e-8, extra_noise_power=irs_noise,
    )
    histories = info.get("history", [])
    objective = [histories[0]["exact_objective_before"]]
    objective.extend(h["exact_objective_after"] for h in histories)
    norm_bdot_sq = np.linalg.norm(exp.ch["b_dot"]) ** 2
    crb = [
        sigma2_s / (2 * T * exp.ch["alpha_sq"] * value * norm_bdot_sq)
        for value in objective
    ]
    return np.arange(len(crb)), np.asarray(crb)


def figure_02_convergence(exp):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for mode, marker in [("passive", "o"), ("active", "s")]:
        iteration, crb = convergence_history(exp, mode)
        ax.plot(
            iteration, crb, marker=marker, color=COLORS[mode],
            label=mode.capitalize(),
        )
    ax.set_xlabel("SCA iteration")
    ax.set_xticks(np.arange(0, max(ax.get_xlim()[1], 2) + 1, 1))
    format_crb_axis(ax, log=True)
    ax.set_title(r"Convergence ($N=40$, $\gamma_0=10$ dB)")
    ax.legend()
    save_figure(fig, "fig02_convergence")


def figure_03_core_tradeoff(exp, gamma_grid):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for mode, label in [
        ("los", "LoS, no IRS"),
        ("passive", "NLoS, Passive IRS"),
        ("active", "NLoS, Active IRS"),
    ]:
        n_irs = 0 if mode == "los" else 40
        results = sweep_gamma(exp, mode, n_irs, gamma_grid)
        x, y = valid_series(results, "rate")
        ax.plot(x, y, color=COLORS[mode], label=label)
    ax.set_xlabel("Communication rate (bps/Hz)")
    format_crb_axis(ax, log=True)
    ax.set_title(r"CRB-rate region at $N=40$")
    ax.legend()
    save_figure(fig, "fig03a_crb_rate_n40")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharey=True)
    for ax, mode in zip(axes, ["passive", "active"]):
        for n_irs in N_irs_list:
            results = sweep_gamma(exp, mode, n_irs, gamma_grid)
            x, y = valid_series(results, "rate")
            ax.plot(x, y, color=N_COLORS[n_irs], label=f"N={n_irs}")
        ax.set_xlabel("Communication rate (bps/Hz)")
        format_crb_axis(ax, log=True)
        ax.set_title(f"{mode.capitalize()} IRS")
        ax.legend()
    save_figure(fig, "fig03b_crb_rate_array_sizes")


def figure_04_sinr_tradeoff(exp, gamma_grid):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for mode, label in [
        ("los", "LoS, no IRS"),
        ("passive", "Passive IRS, N=64"),
        ("active", "Active IRS, N=64"),
    ]:
        n_irs = 0 if mode == "los" else 64
        results = sweep_gamma(exp, mode, n_irs, gamma_grid)
        x, y = valid_series(results, "gamma_db")
        ax.plot(x, y, color=COLORS[mode], label=label)
    ax.set_xlabel(r"SINR threshold $\gamma_0$ (dB)")
    format_crb_axis(ax, log=True)
    ax.set_title("CRB versus communication SINR threshold")
    ax.legend()
    save_figure(fig, "fig04_crb_vs_sinr")


def figure_05_array_size(exp):
    n_grid = [8, 16, 24, 32, 40, 64, 96, 128]
    fig, ax = plt.subplots(figsize=(8.3, 5.3))
    for mode, marker in [("passive", "o"), ("active", "s")]:
        for gamma_db in GAMMA_STYLES:
            crb = [
                exp.solve(mode, gamma_db, n_irs)["crb"]
                for n_irs in n_grid
            ]
            ax.plot(
                n_grid, crb,
                color=COLORS[mode],
                linestyle=GAMMA_STYLES[gamma_db],
                marker=marker,
                markersize=4,
                label=f"{mode.capitalize()}, {gamma_db:g} dB",
            )
    ax.set_xlabel("Number of IRS elements N")
    format_crb_axis(ax, log=True)
    ax.set_title(
        "CRB versus the number of IRS elements under different SINR thresholds"
    )
    ax.legend(ncol=2)
    save_figure(fig, "fig05_crb_vs_array_size")


def figure_06_amplification(exp):
    amax_grid = np.linspace(1.0, 12.0, 12)
    metrics = [
        exp.solve("active", 10.0, 64, amax=float(amax))
        for amax in amax_grid
    ]
    passive = exp.solve("passive", 10.0, 64)["crb"]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(amax_grid, [m["crb"] for m in metrics], "o-",
            color=COLORS["active"], label="Active IRS")
    ax.axhline(passive, color=COLORS["passive"], ls="--",
               label="Passive IRS")
    ax.set_xlabel(r"Maximum amplitude $A_{\max}$")
    format_crb_axis(ax, log=True)
    ax.set_title(r"CRB versus $A_{\max}$ ($N=64$, $\gamma_0=10$ dB)")
    ax.legend()
    save_figure(fig, "fig06a_crb_vs_amax")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    output_dbm = [
        10 * np.log10(m["irs_output_power"]) + 30 for m in metrics
    ]
    ax.plot(amax_grid, output_dbm, "s-", color="#e6550d",
            label="Actual IRS output")
    ax.axhline(P_RIS_dBm, color="black", ls="--",
               label="IRS power budget")
    ax.set_xlabel(r"Maximum amplitude $A_{\max}$")
    ax.set_ylabel("IRS output power (dBm)")
    ax.set_title(r"IRS output power versus $A_{\max}$")
    ax.legend()
    save_figure(fig, "fig06b_irs_power_vs_amax")


def figure_07_ris_power(exp):
    power_dbm = np.linspace(-25, 10, 12)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for n_irs in [32, 64, 128]:
        crb = [
            exp.solve(
                "active", 10.0, n_irs,
                ris_power=10 ** ((value - 30) / 10),
            )["crb"]
            for value in power_dbm
        ]
        ax.plot(power_dbm, crb, marker="o",
                color=N_COLORS[n_irs], label=f"N={n_irs}")
    ax.set_xlabel(r"Active-IRS power budget $P_{\rm RIS}$ (dBm)")
    format_crb_axis(ax, log=True)
    ax.set_title(r"CRB versus Active-IRS power budget ($\gamma_0=10$ dB)")
    ax.legend()
    save_figure(fig, "fig07_crb_vs_ris_power")


def figure_08_noise(exp):
    noise_dbm = np.linspace(-100, -5, 16)
    full = [
        exp.solve(
            "active", 10.0, 64,
            sigma2_irs=10 ** ((value - 30) / 10),
        )
        for value in noise_dbm
    ]
    ideal_crb = exp.solve(
        "active_ideal", 10.0, 64, sigma2_irs=0.0
    )["crb"]
    passive_crb = exp.solve("passive", 10.0, 64)["crb"]

    fig, ax_crb = plt.subplots(figsize=(8.0, 5.2))
    ax_noise = ax_crb.twinx()
    ax_crb.plot(
        noise_dbm, [m["crb"] for m in full],
        color=COLORS["active"], label="Active IRS CRB",
    )
    ax_crb.axhline(
        ideal_crb, color="#ff9896", ls="--", label="Ideal noiseless Active IRS"
    )
    ax_crb.axhline(
        passive_crb, color=COLORS["passive"], ls=":", label="Passive IRS"
    )
    noise_ratio = [m["irs_noise_power"] / sigma2_c for m in full]
    ax_noise.plot(
        noise_dbm, noise_ratio, color=COLORS["noise"],
        marker="s", markersize=3, label="CU IRS-noise / thermal-noise",
    )
    ax_noise.set_yscale("log")
    ax_noise.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    ax_noise.set_ylabel("CU noise-power ratio", color=COLORS["noise"])
    ax_noise.tick_params(axis="y", colors=COLORS["noise"])
    ax_crb.set_xlabel(r"Per-element IRS noise $\sigma_I^2$ (dBm)")
    format_crb_axis(ax_crb, log=True)
    ax_crb.set_title(r"Active-IRS noise sensitivity ($N=64$, $\gamma_0=10$ dB)")
    handles1, labels1 = ax_crb.get_legend_handles_labels()
    handles2, labels2 = ax_noise.get_legend_handles_labels()
    ax_crb.legend(handles1 + handles2, labels1 + labels2, loc="best")
    save_figure(fig, "fig08_crb_and_noise_ratio")


def figure_10_power_fairness(exp, gamma_grid):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    comparisons = [
        ("Same BS power", P, P),
        ("Equal total power", P + P_RIS, P),
    ]
    for ax, (title, passive_power, active_power) in zip(axes, comparisons):
        cases = [
            ("los", 0, passive_power, "LoS, no IRS"),
            ("passive", 40, passive_power, "Passive IRS"),
            ("active", 40, active_power, "Active IRS"),
        ]
        for mode, n_irs, bs_power, label in cases:
            results = sweep_gamma(
                exp, mode, n_irs, gamma_grid, bs_power=bs_power
            )
            x, y = valid_series(results, "rate")
            ax.plot(x, y, color=COLORS[mode], label=label)
        ax.set_xlabel("Communication rate (bps/Hz)")
        format_crb_axis(ax, log=True)
        ax.set_title(title)
        ax.legend()
    save_figure(fig, "fig10_power_accounting")


def render_table(stem, title, columns, rows, widths=None):
    fig_height = max(2.6, 0.48 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=columns, cellLoc="center",
        loc="center", colWidths=widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
    ax.set_title(title, weight="bold", pad=14)
    save_figure(fig, stem)
    with (OUTPUT_DIR / f"{stem}.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def table_01_parameters():
    rows = [
        ["BS antennas", r"$M_t$", Mt, "elements"],
        ["Sensing RX antennas", r"$M_r$", Mr, "elements"],
        ["Coherent samples", r"$T$", T, "symbols"],
        ["BS transmit power", r"$P$", f"{10*np.log10(P)+30:.1f}", "dBm"],
        ["IRS power budget", r"$P_{RIS}$", f"{P_RIS_dBm:.1f}", "dBm"],
        ["Maximum IRS amplitude", r"$A_{max}$", f"{A_MAX:.1f}", "linear"],
        ["IRS noise per element", r"$\sigma_I^2$",
         f"{SIGMA2_RIS_dBm:.1f}", "dBm"],
        ["CU receiver noise", r"$\sigma_c^2$",
         f"{sigma2_c_dBm:.1f}", "dBm"],
        ["Sensing RX noise", r"$\sigma_s^2$",
         f"{sigma2_s_dBm:.1f}", "dBm"],
        ["Path-loss reference", r"$K_0$", f"{K0:.1f}", "dB"],
        ["Path-loss exponent", r"$\alpha_0$", f"{alpha0:.1f}", "-"],
        ["Rician factor", r"$K_c$", f"{Kc:.1f}", "linear"],
        ["IRS elements", r"$N$", ", ".join(map(str, N_irs_list)), "elements"],
        ["SINR sweep", r"$\gamma_0$",
         f"{gamma_0_dB_min:.0f} to {gamma_0_dB_max:.0f}", "dB"],
        ["Direct / IRS-CU seeds", "-", f"{SEED_CHANNEL} / {SEED_IRS_CU}", "-"],
    ]
    render_table(
        "table01_simulation_parameters",
        "Table 1. Simulation parameters",
        ["Parameter", "Symbol", "Value", "Unit"],
        rows,
        widths=[0.35, 0.18, 0.25, 0.15],
    )


def table_03_core_results(exp):
    rows = []
    cases = [
        ("LoS, no IRS", "los", 0),
        ("NLoS, Passive IRS", "passive", 64),
        ("NLoS, Active IRS", "active", 64),
    ]
    for label, mode, n_irs in cases:
        metric = exp.solve(mode, 10.0, n_irs)
        output = (
            f"{10*np.log10(metric['irs_output_power'])+30:.2f}"
            if metric["irs_output_power"] > 0 else "-"
        )
        gain = f"{metric['gain']:.2f}" if metric["gain"] > 0 else "-"
        rows.append([
            label,
            f"{metric['crb']:.4e}",
            f"{metric['rate']:.4f}",
            output,
            gain,
            f"{metric['runtime_s']:.3f}",
        ])
    render_table(
        "table03_core_results",
        r"Table 3. Core results ($N=64$, $\gamma_0=10$ dB)",
        ["Scheme", r"CRB (rad$^2$)", "Rate (bps/Hz)",
         "IRS power (dBm)", "Max gain", "Runtime (s)"],
        rows,
        widths=[0.28, 0.16, 0.16, 0.16, 0.12, 0.12],
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    exp = Experiment()
    gamma_grid = np.array(
        [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 19],
        dtype=float,
    )

    figure_01_system_model()
    figure_02_convergence(exp)
    figure_03_core_tradeoff(exp, gamma_grid)
    figure_04_sinr_tradeoff(exp, gamma_grid)
    figure_05_array_size(exp)
    figure_06_amplification(exp)
    figure_07_ris_power(exp)
    figure_08_noise(exp)
    figure_10_power_fairness(exp, gamma_grid)
    table_01_parameters()
    table_03_core_results(exp)
    print(f"All paper figures and tables saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
