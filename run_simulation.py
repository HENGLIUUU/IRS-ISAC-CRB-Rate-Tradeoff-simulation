"""
IRS-assisted Bistatic ISAC — CRB-Rate Tradeoff with Active IRS
=============================================================
对比 Passive vs Active IRS 在 NLoS 感知中的性能。
Active IRS:   |v[n]| ≤ A_MAX（可放大）
Passive IRS:  |v[n]| = 1（仅调相位）

用法:
    python run_simulation.py
"""

import os
import sys
import time
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

from config import (
    Mt, Mr, T, P_dBm, SEED, N_gamma, N_irs_list, pos_irs, A_MAX,
)
from channel_constant import init_geometry_and_channels, generate_irs_channels
from scenario import scan_scenario
from plot_irs import plot_irs_comparison


def main():
    print("=" * 60, flush=True)
    print("Active vs Passive IRS for Bistatic ISAC", flush=True)
    print("=" * 60, flush=True)
    np.random.seed(SEED)

    # ---- Initialize ----
    geo, ch = init_geometry_and_channels()
    irs_ch = {N: generate_irs_channels(N, pos_irs, geo) for N in N_irs_list}

    print(f"\nMt={Mt}, Mr={Mr}, T={T}, P={P_dBm} dBm", flush=True)
    print(f"IRS at ({pos_irs[0]:.0f}, {pos_irs[1]:.0f})", flush=True)
    print(f"Active IRS: A_MAX = {A_MAX} ({10*np.log10(A_MAX):.1f} dB)", flush=True)
    print(f"|alpha|^2 = {ch['alpha_sq']:.3e}", flush=True)
    print(f"N values: {N_irs_list}", flush=True)

    # ---- Scenario table ----
    # Format: (label, use_irs, N_irs, target_blocked, active, npts)
    scenarios = [
        # ── LoS baseline ──
        ("LoS (no IRS)",          False, 0,   False, False, N_gamma),

        # ── Passive IRS (NLoS) ──
        ("NLoS+IRS N=16 (P)",     True,  16,  True,  False, N_gamma),
        ("NLoS+IRS N=32 (P)",     True,  32,  True,  False, N_gamma),
        ("NLoS+IRS N=64 (P)",     True,  64,  True,  False, N_gamma),
        ("NLoS+IRS N=128 (P)",    True,  128, True,  False, N_gamma),

        # ── Active IRS (NLoS) ──
        ("NLoS+IRS N=16 (A)",     True,  16,  True,  True,  N_gamma),
        ("NLoS+IRS N=32 (A)",     True,  32,  True,  True,  N_gamma),
        ("NLoS+IRS N=64 (A)",     True,  64,  True,  True,  N_gamma),
        ("NLoS+IRS N=128 (A)",    True,  128, True,  True,  N_gamma),
    ]

    all_data = {}
    for label, use_irs, n_val, blocked, active, npts in scenarios:
        irs = irs_ch.get(n_val) if use_irs else None
        data = scan_scenario(label, use_irs, n_val, geo, ch, irs,
                             direct_blocked=blocked, active=active, npts_override=npts)
        all_data[label] = data

    # ---- Save data ----
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    save_dict = {}
    for label, data in all_data.items():
        if data is not None:
            key = label.replace(", ", "_").replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
            save_dict[f"{key}_gamma"] = data["gamma"]
            save_dict[f"{key}_crb"] = data["crb"]
            save_dict[f"{key}_rate"] = data["rate"]

    data_path = os.path.join(out_dir, 'crb_vs_sinr_data.npz')
    np.savez(data_path, **save_dict)
    print(f"\nData saved to: {data_path}", flush=True)

    baseline = all_data.get("LoS (no IRS)")
    plot_keys = [
        "NLoS+IRS N=16 (P)", "NLoS+IRS N=32 (P)", "NLoS+IRS N=64 (P)", "NLoS+IRS N=128 (P)",
        "NLoS+IRS N=16 (A)", "NLoS+IRS N=32 (A)", "NLoS+IRS N=64 (A)", "NLoS+IRS N=128 (A)",
    ]
    curves = [(all_data[k], k) for k in plot_keys if all_data.get(k) is not None]

    plot_irs_comparison(
        baseline, [c for c, _ in curves], labels=[l for _, l in curves],
        save_path=os.path.join(out_dir, 'crb_vs_sinr.png'),
        use_log=True
    )
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
