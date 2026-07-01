"""
IRS-assisted Bistatic ISAC — CRB-Rate Tradeoff with IRS Enhancement
===================================================================
基于 CRB-Rate Tradeoff for Bistatic ISAC (TWC 2026) 扩展 IRS 辅助版本。
信号模型: Case 2 (叠加信号: 高斯信息 + 确定性感知信号)

用法:
    python run_simulation.py
"""

import os
import sys
import time
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

from config import (
    Mt, Mr, T, P_dBm, SEED, N_gamma, N_irs_list, pos_irs,
)
from channel_constant import init_geometry_and_channels, generate_irs_channels
from scenario import scan_scenario
from plot_irs import plot_irs_comparison


def main():
    print("=" * 60, flush=True)
    print("IRS-assisted Bistatic ISAC -- Scenario Comparison", flush=True)
    print("=" * 60, flush=True)
    np.random.seed(SEED)

    # ---- Initialize shared geometry & channels ----
    geo, ch = init_geometry_and_channels()

    # Pre-generate IRS channels for all N values
    irs_ch = {N: generate_irs_channels(N, pos_irs, geo) for N in N_irs_list}

    print(f"\nMt={Mt}, Mr={Mr}, T={T}, P={P_dBm} dBm", flush=True)
    print(f"IRS at ({pos_irs[0]:.0f}, {pos_irs[1]:.0f})", flush=True)
    print(f"|alpha|^2 = {ch['alpha_sq']:.3e}", flush=True)
    print(f"N values: {N_irs_list}", flush=True)

    # ---- Scenario table: add new scenarios by appending rows ----
    # Format: (label, use_irs, N_irs, target_blocked, npts)
    scenarios = [
        # LoS baseline
        ("LoS (no IRS)",              False, 0,   False, N_gamma),

        # NLoS scenarios
        ("NLoS no IRS",            False, 0,   True,  1),
        ("NLoS+IRS N=16",          True,  16,  True,  N_gamma),
        ("NLoS+IRS N=32",          True,  32,  True,  N_gamma),
        ("NLoS+IRS N=64",          True,  64,  True,  N_gamma),
        ("NLoS+IRS N=128",         True,  128, True,  N_gamma),

        # LoS + IRS (full AO, for reference)
        ("LoS+IRS N=32",           True,  32,  False, N_gamma),
    ]

    all_data = {}
    for label, use_irs, n_val, blocked, npts in scenarios:
        irs = irs_ch.get(n_val) if use_irs else None
        data = scan_scenario(label, use_irs, n_val, geo, ch, irs,
                             direct_blocked=blocked, npts_override=npts)
        all_data[label] = data

    # ---- Save data ----
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    save_dict = {}
    for label, data in all_data.items():
        if data is not None:
            key = label.replace(", ", "_").replace(" ", "_").replace("+", "_")
            save_dict[f"{key}_gamma"] = data["gamma"]
            save_dict[f"{key}_crb"] = data["crb"]
            save_dict[f"{key}_rate"] = data["rate"]

    data_path = os.path.join(out_dir, f'irs_comparison_all_{timestamp}.npz')
    np.savez(data_path, **save_dict)
    print(f"\nData saved to: {data_path}", flush=True)

    # ---- Unified comparison plot ----
    baseline = all_data.get("LoS (no IRS)")
    plot_keys = ["NLoS+IRS N=16", "NLoS+IRS N=32", "NLoS+IRS N=64", "NLoS+IRS N=128", "LoS+IRS N=32"]
    curves = [(all_data[k], k) for k in plot_keys if all_data.get(k) is not None]

    plot_irs_comparison(
        baseline, [c for c, _ in curves], labels=[l for _, l in curves],
        save_path=os.path.join(out_dir, f'comparison_all_{timestamp}.png'),
        use_log=True
    )
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
