"""
IRS-assisted Bistatic ISAC — AO Framework Main Script
===================================================
基于 Paper 4 (CRB-Rate Tradeoff, TWC 2026) 扩展 IRS 辅助版本。
信号模型: Case 2 (叠加信号: 高斯信息 + 确定性感知信号)

AO 框架:
  Step 1: 固定 Theta, 优化 R_c, R_s (复用 sca_solver.py)
  Step 2: 固定 R_c, R_s, 优化 Theta (irs_solver.py SDR)

场景对比:
  - LoS: BS->Target 直射径通畅
  - NLoS: BS->Target 直射径遮挡, 仅靠 IRS 反射径感知
  - IRS 规模: N=16, N=32

用法:
    python run_simulation.py
"""

import os
import time
import numpy as np

from config import (
    Mt, Mr, T, Kc, K0, alpha0, d0, CAL_ALPHA,
    N_gamma, gamma_0_dB_min, gamma_0_dB_max,
    pos_bs, pos_target, pos_rx, pos_irs,
    phi_target, theta_target, phi_cu,
    sigma2_c, sigma2_s, P, P_dBm, SEED, SEED_CHANNEL,
    AO_MAX_ITER, AO_TOL, SDR_TRIALS,
)
from steering_vectors import steering_vector, steering_vector_derivative
from channels import (
    generate_rician_channel, compute_alpha_sq,
    generate_irs_bs_channel, generate_irs_target_channel,
    generate_irs_cu_channel,
    compute_effective_a, compute_effective_h, irs_beam_align,
    compute_distance, compute_angle,
)
from crb import compute_crb_irs
from rate import compute_rate_irs
from sca_solver import solve_p4_sca
from irs_solver import solve_irs_sdr
from plot_irs import plot_irs_comparison


# ========================================================================
# 信道和几何初始化（所有场景共享）
# ========================================================================
def init_geometry_and_channels():
    """Compute distances/angles and generate all channels once."""
    d_bt = compute_distance(pos_bs, pos_target)
    d_tr = compute_distance(pos_target, pos_rx)
    d_br = compute_distance(pos_bs, pos_irs)
    d_rt = compute_distance(pos_irs, pos_target)

    pos_cu = np.array([1000.0 * np.cos(phi_cu), 1000.0 * np.sin(phi_cu)])
    d_rc = compute_distance(pos_irs, pos_cu)
    d_bc = compute_distance(pos_bs, pos_cu)

    phi_br = compute_angle(pos_bs, pos_irs)
    phi_rt = compute_angle(pos_irs, pos_target)
    phi_rc = compute_angle(pos_irs, pos_cu)

    h = generate_rician_channel(Mt, phi_cu, Kc, d_bc,
                                K0, alpha0, d0, SEED_CHANNEL)
    a_dir = steering_vector(Mt, phi_target)
    b = steering_vector(Mr, theta_target)
    b_dot = steering_vector_derivative(Mr, theta_target)
    alpha_sq = compute_alpha_sq(d_bt, d_tr, 1.0,
                                K0, alpha0, d0, CAL_ALPHA)

    geo = {
        "d_bt": d_bt, "d_tr": d_tr, "d_br": d_br, "d_rt": d_rt,
        "d_rc": d_rc, "d_bc": d_bc,
        "phi_br": phi_br, "phi_rt": phi_rt, "phi_rc": phi_rc,
        "pos_cu": pos_cu,
    }
    channels = {
        "h": h, "a_dir": a_dir, "b": b, "b_dot": b_dot,
        "alpha_sq": alpha_sq,
    }
    return geo, channels


def generate_irs_channels(N_irs, geo):
    """Generate IRS channels for a given N_irs."""
    G = generate_irs_bs_channel(Mt, N_irs, geo["d_br"], geo["phi_br"],
                                K0, alpha0, d0)
    h_r = generate_irs_target_channel(N_irs, geo["d_rt"], geo["phi_rt"],
                                       K0, alpha0, d0)
    h_rc = generate_irs_cu_channel(N_irs, geo["d_rc"], geo["phi_rc"], Kc,
                                    K0, alpha0, d0, SEED_CHANNEL)
    return G, h_r, h_rc


# ========================================================================
# AO 优化（单 gamma_0 点）
# ========================================================================
def ao_optimize(gamma_0, a_eff_init, h_eff_init, G, h_r, h_rc,
                b, b_dot, alpha_sq,
                sigma2_c, sigma2_s, P, Mt, Mr, T, N_irs,
                direct_blocked=False, a_dir=None):
    """
    AO: Alternating optimization for IRS-assisted ISAC.

    Iterates between:
      Step 1: Fix Theta, optimize R_c, R_s (sca_solver)
      Step 2: Fix R_c, R_s, optimize Theta (irs_solver)
    """
    v = np.ones(N_irs, dtype=complex)
    a_eff = a_eff_init.copy()
    h_eff = h_eff_init.copy()

    history = []

    for k in range(AO_MAX_ITER):
        # Step 1: Fix Theta, optimize R_c, R_s
        Rc, Rs, info_sca = solve_p4_sca(
            gamma_0, h_eff, a_eff, sigma2_c, sigma2_s,
            P, Mt, Mr, b, b_dot, alpha_sq
        )
        if Rc is None:
            return None, None, None, {"status": f"SCA failed at AO iter {k}"}

        # Step 2: Fix R_c, R_s, optimize Theta
        a_sdr = a_dir if a_dir is not None and not direct_blocked else np.zeros(Mt, dtype=complex)
        v_new, info_irs = solve_irs_sdr(
            Rc, Rs, a_sdr, h_eff, G, h_r, h_rc, b, b_dot, alpha_sq,
            sigma2_c, sigma2_s, gamma_0, T, N_irs, Mt, trials=SDR_TRIALS
        )
        if v_new is None:
            v_new = v  # SDR failed -- keep current

        # Update effective channels
        a_eff_new = compute_effective_a(
            a_dir if a_dir is not None else a_eff,
            G, h_r, v_new, direct_blocked=direct_blocked
        )
        h_eff_new = compute_effective_h(h_eff, G, h_rc, v_new)

        # Convergence check
        crb_old = compute_crb_irs(Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)
        crb_new = compute_crb_irs(Rc, Rs, a_eff_new, b, b_dot, alpha_sq, sigma2_s, T)

        history.append({"iter": k, "crb": crb_new})
        crb_change = abs(crb_new - crb_old) / (abs(crb_old) + 1e-15)
        v, a_eff, h_eff = v_new, a_eff_new, h_eff_new

        if crb_change < AO_TOL and k > 0:
            break

    return Rc, Rs, v, {"status": f"converged in {k+1} AO iters", "history": history}


# ========================================================================
# 场景扫描（每个场景对应一种函数）
# ========================================================================
def scan_baseline(gamma_0, h, a_dir, b, b_dot, alpha_sq):
    """
    Baseline: no IRS, direct LoS for both sensing and communication.
    Uses SCA directly (no IRS means no alternating optimization needed).
    """
    Rc, Rs, info = solve_p4_sca(
        gamma_0, h, a_dir, sigma2_c, sigma2_s,
        P, Mt, Mr, b, b_dot, alpha_sq
    )
    if Rc is None:
        return None, None, None, None
    return Rc, Rs, a_dir, h


def scan_nlos_with_irs(gamma_0, a_dir, h, G, h_r, h_rc, b, b_dot, alpha_sq):
    """
    NLoS + IRS: direct path blocked, only IRS reflection path.
    No AO needed — beam-align IRS phases (closed form), then single SCA.
    """
    v = irs_beam_align(h_r, G)
    a_eff = compute_effective_a(a_dir, G, h_r, v, direct_blocked=True)
    h_eff = compute_effective_h(h, G, h_rc, v)

    Rc, Rs, info = solve_p4_sca(
        gamma_0, h_eff, a_eff, sigma2_c, sigma2_s,
        P, Mt, Mr, b, b_dot, alpha_sq
    )
    if Rc is None:
        return None, None, None, None
    return Rc, Rs, a_eff, h_eff


def scan_los_with_irs(gamma_0, a_dir, h, G, h_r, h_rc, b, b_dot, alpha_sq, N_irs):
    """
    LoS + IRS: both direct and IRS paths exist.
    Full AO (alternating between SCA beamforming and SDR IRS optimization).
    """
    a_eff_init = compute_effective_a(a_dir, G, h_r, np.ones(N_irs, dtype=complex))
    h_eff_init = compute_effective_h(h, G, h_rc, np.ones(N_irs, dtype=complex))

    Rc, Rs, theta_opt, info = ao_optimize(
        gamma_0, a_eff_init, h_eff_init, G, h_r, h_rc,
        b, b_dot, alpha_sq, sigma2_c, sigma2_s,
        P, Mt, Mr, T, N_irs,
        direct_blocked=False, a_dir=a_dir
    )
    if Rc is None:
        return None, None, None, None

    a_eff = compute_effective_a(a_dir, G, h_r, theta_opt)
    h_eff = compute_effective_h(h, G, h_rc, theta_opt)
    return Rc, Rs, a_eff, h_eff


def scan_scenario(label, use_irs, N_irs_scan, geo, ch, irs_ch,
                  direct_blocked=False, npts_override=None):
    """Run SINR sweep over gamma_0 for one scenario."""
    print(f"\n--- Scenario: {label} ---")

    if use_irs:
        G, h_r, h_rc = irs_ch
    else:
        G, h_r, h_rc = None, None, None

    h = ch["h"]
    a_dir = ch["a_dir"]
    b = ch["b"]
    b_dot = ch["b_dot"]
    alpha_sq = ch["alpha_sq"]

    npts = npts_override if npts_override else N_gamma
    gamma_0_dB_vals = np.linspace(gamma_0_dB_min, gamma_0_dB_max, npts)
    results = []

    for g0_dB in gamma_0_dB_vals:
        gamma_0 = 10**(g0_dB / 10)

        if direct_blocked and not use_irs:
            # NLoS without IRS: no sensing path at all
            results.append((gamma_0, None, None))
            continue
        elif use_irs and direct_blocked:
            Rc, Rs, a_eff, h_eff = scan_nlos_with_irs(gamma_0, a_dir, h, G, h_r, h_rc, b, b_dot, alpha_sq)
        elif use_irs and not direct_blocked:
            Rc, Rs, a_eff, h_eff = scan_los_with_irs(gamma_0, a_dir, h, G, h_r, h_rc, b, b_dot, alpha_sq, N_irs_scan)
        else:
            Rc, Rs, a_eff, h_eff = scan_baseline(gamma_0, h, a_dir, b, b_dot, alpha_sq)

        if Rc is None:
            results.append((gamma_0, None, None))
            continue

        rate, sinr = compute_rate_irs(Rc, Rs, h_eff, sigma2_c)
        crb = compute_crb_irs(Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)
        results.append((gamma_0, crb, rate))

    # Parse results
    valid = [(g, c, r) for g, c, r in results if c is not None]
    if len(valid) < 3:
        print(f"  {label}: too few feasible points ({len(valid)})")
        return None

    g_arr, c_arr, r_arr = zip(*valid)
    data = {"gamma": np.array(g_arr), "crb": np.array(c_arr), "rate": np.array(r_arr)}
    print(f"  Feasible points: {len(valid)}/{npts}, "
          f"CRB range: [{c_arr[-1]:.3e}, {c_arr[0]:.3e}]")
    return data


# ========================================================================
# 主程序
# ========================================================================
def main():
    print("=" * 60)
    print("IRS-assisted Bistatic ISAC -- Scenario Comparison")
    print("=" * 60)
    np.random.seed(SEED)

    geo, ch = init_geometry_and_channels()
    irs_ch_16 = generate_irs_channels(16, geo)
    irs_ch_32 = generate_irs_channels(32, geo)

    print(f"\nMt={Mt}, Mr={Mr}, T={T}, P={P_dBm} dBm")
    print(f"IRS at ({pos_irs[0]:.0f}, {pos_irs[1]:.0f})")
    print(f"d_br={geo['d_br']:.1f}m, d_rt={geo['d_rt']:.1f}m")
    print(f"|alpha|^2 = {ch['alpha_sq']:.3e}")

    nlos_N_gamma = max(10, N_gamma // 4)

    scenarios = [
        ("LoS, no IRS",     False, 0,   False, N_gamma),
        ("NLoS, no IRS",    False, 0,   True,  1),
        ("NLoS, IRS N=16",  True,  16,  True,  nlos_N_gamma),
        ("NLoS, IRS N=32",  True,  32,  True,  nlos_N_gamma),
        ("LoS, IRS N=32",   True,  32,  False, N_gamma),
    ]

    all_data = {}
    for label, use_irs, n_val, blocked, npts in scenarios:
        data = scan_scenario(label, use_irs, n_val, geo, ch,
                             irs_ch_16 if n_val == 16 else irs_ch_32,
                             direct_blocked=blocked, npts_override=npts)
        all_data[label] = data

    # Save data
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    save_dict = {}
    for label, data in all_data.items():
        if data is not None:
            key = label.replace(", ", "_").replace(" ", "_")
            save_dict[f"{key}_gamma"] = data["gamma"]
            save_dict[f"{key}_crb"] = data["crb"]
            save_dict[f"{key}_rate"] = data["rate"]

    data_path = os.path.join(out_dir, f'irs_comparison_all_{timestamp}.npz')
    np.savez(data_path, **save_dict)
    print(f"\nData saved to: {data_path}")

    # Plot
    irs_curves = []
    irs_labels = []
    for lbl in ["NLoS, IRS N=16", "NLoS, IRS N=32", "LoS, IRS N=32"]:
        if all_data.get(lbl) is not None:
            irs_curves.append(all_data[lbl])
            irs_labels.append(lbl)

    plot_irs_comparison(
        all_data.get("LoS, no IRS"),
        irs_curves,
        labels=irs_labels,
        save_path=os.path.join(out_dir, f'comparison_all_{timestamp}.png'),
        use_log=True
    )

    print("Done.")


if __name__ == '__main__':
    main()
