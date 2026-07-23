"""Run the main single-pass Passive/Active IRS bistatic-ISAC sweep."""

import os
import sys
import time
import json
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

from config import (
    Mt, Mr, T, P, P_RIS, P_dBm, P_RIS_dBm, SIGMA2_RIS_dBm,
    SEED, SEED_CHANNEL, SEED_IRS_CU,
    N_gamma, N_irs_list, pos_irs, A_MAX,
    MAIN_IRS_STRATEGY, POWER_ACCOUNTING,
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
    # A_MAX is an amplitude gain, so its dB value uses 20*log10(.).
    print(f"Active IRS: A_MAX = {A_MAX} ({20*np.log10(A_MAX):.1f} dB amplitude)", flush=True)
    print(f"|alpha|^2 = {ch['alpha_sq']:.3e}", flush=True)
    print(f"N values: {N_irs_list}", flush=True)
    print(f"IRS strategy: {MAIN_IRS_STRATEGY}", flush=True)
    print(f"Power accounting: {POWER_ACCOUNTING}", flush=True)

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
    scenario_bs_powers = {}
    for label, use_irs, n_val, blocked, active, npts in scenarios:
        irs = irs_ch.get(n_val) if use_irs else None
        if POWER_ACCOUNTING == "same_bs_power":
            bs_power = P
        elif POWER_ACCOUNTING == "equal_total_power":
            bs_power = P if active else P + P_RIS
        else:
            raise ValueError(
                "POWER_ACCOUNTING must be 'same_bs_power' or "
                "'equal_total_power'."
            )
        data = scan_scenario(label, use_irs, n_val, geo, ch, irs,
                             direct_blocked=blocked, active=active,
                             npts_override=npts,
                             irs_strategy=MAIN_IRS_STRATEGY,
                             bs_power_override=bs_power)
        all_data[label] = data
        scenario_bs_powers[label] = bs_power

    # ---- Save data ----
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(
        os.path.dirname(__file__), 'results', 'runs', timestamp
    )
    os.makedirs(out_dir, exist_ok=True)

    save_dict = {}
    for label, data in all_data.items():
        if data is not None:
            key = label.replace(", ", "_").replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
            for metric, values in data.items():
                if isinstance(values, np.ndarray):
                    save_dict[f"{key}_{metric}"] = values

    data_path = os.path.join(out_dir, 'crb_vs_sinr_data.npz')
    np.savez(data_path, **save_dict)
    print(f"\nData saved to: {data_path}", flush=True)

    metadata = {
        "created_at": timestamp,
        "model": (
            "CRB-Rate Tradeoff for Bistatic ISAC with a simplified "
            "single-forward-pass IRS extension"
        ),
        "Mt": Mt,
        "Mr": Mr,
        "T": T,
        "P_dBm": P_dBm,
        "A_MAX": A_MAX,
        "P_RIS_dBm": P_RIS_dBm,
        "SIGMA2_RIS_dBm_per_element": SIGMA2_RIS_dBm,
        "irs_strategy": MAIN_IRS_STRATEGY,
        "power_accounting": POWER_ACCOUNTING,
        "scenario_bs_power_W": scenario_bs_powers,
        "sensing_noise_model": (
            "White receiver noise from CRB-Rate Tradeoff for Bistatic ISAC; "
            "first-pass Active-IRS noise "
            "is saved as a negligible diagnostic but omitted from the CRB"
        ),
        "N_irs_list": list(N_irs_list),
        "N_gamma": N_gamma,
        "seed": SEED,
        "seed_direct_cu": SEED_CHANNEL,
        "seed_irs_cu": SEED_IRS_CU,
    }
    with open(
        os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8"
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    baseline = all_data.get("LoS (no IRS)")
    plot_keys = [
        "NLoS+IRS N=16 (P)", "NLoS+IRS N=32 (P)", "NLoS+IRS N=64 (P)", "NLoS+IRS N=128 (P)",
        "NLoS+IRS N=16 (A)", "NLoS+IRS N=32 (A)", "NLoS+IRS N=64 (A)", "NLoS+IRS N=128 (A)",
    ]
    curves = [(all_data[k], k) for k in plot_keys if all_data.get(k) is not None]

    plot_irs_comparison(
        baseline, [c for c, _ in curves], labels=[l for _, l in curves],
        save_path=os.path.join(out_dir, 'crb_vs_sinr.png'),
        use_log=True,
        subtitle=(
            f"{MAIN_IRS_STRATEGY} IRS baseline; "
            + (
                "same BS power"
                if POWER_ACCOUNTING == "same_bs_power"
                else "equal total power"
            )
        ),
    )
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
