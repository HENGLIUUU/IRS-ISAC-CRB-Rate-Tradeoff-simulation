"""
IRS-assisted Bistatic ISAC — AO Framework Main Script
===================================================
基于 Paper 4 (CRB-Rate Tradeoff, TWC 2026) 扩展 IRS 辅助版本。
信号模型: Case 2 (叠加信号: 高斯信息 + 确定性感知信号)

AO 框架:
  Step 1: 固定 Θ, 优化 R_c, R_s (复用 case2_solver.py)
  Step 2: 固定 R_c, R_s, 优化 Θ (irs_solver.py SDR)

场景对比:
  - LoS: BS→Target 直射径通畅
  - NLoS: BS→Target 直射径遮挡, 仅靠 IRS 反射径感知
  - IRS 规模: N=16, N=32

用法:
    python main_irs.py
"""

import os, time, numpy as np
import sys

from config import *
from steering_vectors import steering_vector, steering_vector_derivative
from channels import (
    generate_rician_channel, compute_alpha_sq,
    generate_irs_bs_channel, generate_irs_target_channel,
    generate_irs_cu_channel,
    compute_effective_a, compute_effective_h,
    compute_distance, compute_angle
)
from crb_calc import compute_crb_irs
from comm_rate import compute_rate_irs
from case2_solver import solve_p4_sca
from irs_solver import solve_irs_sdr
from plot_results import plot_irs_comparison, plot_comparison


# ========================================================================
# 信道和几何初始化（所有场景共享）
# ========================================================================
def init_geometry_and_channels():
    """Compute distances/angles and generate all channels once."""
    d_bt = compute_distance(pos_bs, pos_target)
    d_tr = compute_distance(pos_target, pos_rx)
    d_br = compute_distance(pos_bs, pos_irs)
    d_rt = compute_distance(pos_irs, pos_target)

    pos_cu = np.array([1000.0 * np.cos(phi_cu),
                       1000.0 * np.sin(phi_cu)])
    d_rc = compute_distance(pos_irs, pos_cu)
    d_bc = compute_distance(pos_bs, pos_cu)

    phi_br = compute_angle(pos_bs, pos_irs)
    phi_rt = compute_angle(pos_irs, pos_target)
    phi_rc = compute_angle(pos_irs, pos_cu)

    # Channels (independent of N_irs — regenerate per scenario)
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
      Step 1: Fix Θ, optimize R_c, R_s (case2_solver)
      Step 2: Fix R_c, R_s, optimize Θ (irs_solver)

    Args:
        direct_blocked: If True, use IRS-only steering in compute_effective_a
        a_dir: Direct-path steering vector (needed for compute_effective_a)

    Returns:
        Rc_opt, Rs_opt, theta_opt, info
    """
    v = np.ones(N_irs, dtype=complex)
    a_eff = a_eff_init.copy()
    h_eff = h_eff_init.copy()

    history = []

    for k in range(AO_MAX_ITER):
        # Step 1: Fix Θ, optimize R_c, R_s
        Rc, Rs, info_sca = solve_p4_sca(
            gamma_0, h_eff, a_eff, sigma2_c, sigma2_s,
            P, Mt, Mr, b, b_dot, alpha_sq
        )
        if Rc is None:
            return None, None, None, {"status": f"SCA failed at AO iter {k}"}

        # Step 2: Fix R_c, R_s, optimize Θ
        # In NLoS: pass zeros for direct-path a in SDR (no linear coupling)
        a_sdr = a_dir if a_dir is not None and not direct_blocked else np.zeros(Mt, dtype=complex)
        v_new, info_irs = solve_irs_sdr(
            Rc, Rs, a_sdr, h_eff, G, h_r, h_rc, b, b_dot, alpha_sq,
            sigma2_c, sigma2_s, gamma_0, T, N_irs, Mt, trials=SDR_TRIALS
        )
        if v_new is None:
            v_new = v  # SDR failed — keep current

        # Update effective channels
        a_eff_new = compute_effective_a(
            a_dir if a_dir is not None else a_eff,
            G, h_r, v_new, direct_blocked=direct_blocked
        )
        h_eff_new = compute_effective_h(h_eff, G, h_rc, v_new)

        # Convergence check
        crb_old = compute_crb_irs(0, Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)
        crb_new = compute_crb_irs(0, Rc, Rs, a_eff_new, b, b_dot, alpha_sq, sigma2_s, T)

        history.append({"iter": k, "crb": crb_new})
        crb_change = abs(crb_new - crb_old) / (abs(crb_old) + 1e-15)
        v, a_eff, h_eff = v_new, a_eff_new, h_eff_new

        if crb_change < AO_TOL and k > 0:
            break

    return Rc, Rs, v, {"status": f"converged in {k+1} AO iters", "history": history}


# ========================================================================
# 单场景扫描
# ========================================================================
def irs_beam_align(h_r, G):
    """
    Align IRS phases to maximize signal toward target direction.

    Sets each element's phase to compensate the BS→IRS and IRS→Target paths,
    making all N reflected signals add coherently at the target.

    v_align[n] = exp(-j * (angle(G[n,0]) + angle(h_r[0,n])))

    Args:
        h_r: IRS→Target channel (1×N)
        G: BS→IRS channel (N×Mt)

    Returns:
        v_align: Phase shift vector (N,) with |v| = 1
    """
    G_ref = G[:, 0]  # use first BS antenna as phase reference
    v = np.exp(-1j * (np.angle(h_r.flatten()) + np.angle(G_ref)))
    return v / (np.abs(v) + 1e-15)


def scan_scenario(label, use_irs, N_irs_scan, geo, ch, irs_ch,
                  direct_blocked=False, npts_override=None):
    """
    Run SINR sweep for one scenario.

    Args:
        use_irs: If True, optimize IRS phases via AO
        N_irs_scan: IRS element count (ignored if use_irs=False)
        direct_blocked: If True, BS→Target LoS is blocked
    """
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

    # NLoS with no IRS — print expected result
    if direct_blocked and not use_irs:
        print(f"  (Direct path blocked, no IRS — no sensing path, expected 0 feasible)")
        results.append((0.0, None, None))

    use_irst = use_irs
    for g0_dB in gamma_0_dB_vals:
        gamma_0 = 10**(g0_dB / 10)

        if use_irst:
            if direct_blocked:
                # NLoS: IRS beam-align + SCA (fast, no AO iteration)
                v = irs_beam_align(h_r, G)
                a_eff = compute_effective_a(a_dir, G, h_r, v,
                                            direct_blocked=True)
                h_eff = compute_effective_h(h, G, h_rc, v)
                Rc, Rs, info = solve_p4_sca(
                    gamma_0, h_eff, a_eff, sigma2_c, sigma2_s,
                    P, Mt, Mr, b, b_dot, alpha_sq
                )
                if Rc is None:
                    results.append((gamma_0, None, None))
                    continue
            else:
                # LoS: full AO
                a_eff_init = compute_effective_a(a_dir, G, h_r,
                                                  np.ones(N_irs_scan, dtype=complex))
                h_eff_init = compute_effective_h(h, G, h_rc,
                                                  np.ones(N_irs_scan, dtype=complex))
                Rc, Rs, theta_opt, info = ao_optimize(
                    gamma_0, a_eff_init, h_eff_init, G, h_r, h_rc,
                    b, b_dot, alpha_sq, sigma2_c, sigma2_s,
                    P, Mt, Mr, T, N_irs_scan,
                    direct_blocked=False, a_dir=a_dir
                )
                if Rc is None:
                    results.append((gamma_0, None, None))
                    continue
                a_eff = compute_effective_a(a_dir, G, h_r, theta_opt)
                h_eff = compute_effective_h(h, G, h_rc, theta_opt)
        else:
            # No IRS — baseline
            if direct_blocked:
                # NLoS without IRS = no sensing path → infeasible
                results.append((gamma_0, None, None))
                continue

            Rc, Rs, info = solve_p4_sca(
                gamma_0, h, a_dir, sigma2_c, sigma2_s,
                P, Mt, Mr, b, b_dot, alpha_sq
            )
            if Rc is None:
                results.append((gamma_0, None, None))
                continue

            a_eff = a_dir
            h_eff = h

        rate, sinr = compute_rate_irs(Rc, Rs, h_eff, sigma2_c)
        crb = compute_crb_irs(0, Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)
        results.append((gamma_0, crb, rate))

    # Parse results
    valid = [(g, c, r) for g, c, r in results if c is not None]
    if len(valid) < 3:
        print(f"  {label}: too few feasible points ({len(valid)})")
        return None

    g_arr, c_arr, r_arr = zip(*valid)
    data = {"gamma": np.array(g_arr), "crb": np.array(c_arr), "rate": np.array(r_arr)}
    print(f"  Feasible points: {len(valid)}/{N_gamma}, "
          f"CRB range: [{c_arr[-1]:.3e}, {c_arr[0]:.3e}]")
    return data


# ========================================================================
# 主程序
# ========================================================================
def main():
    print("=" * 60)
    print("IRS-assisted Bistatic ISAC — Scenario Comparison")
    print("=" * 60)
    np.random.seed(SEED)

    # Init all shared geometry and channels
    geo, ch = init_geometry_and_channels()

    # Init IRS channels for N=16, N=32
    irs_ch_16 = generate_irs_channels(16, geo)
    irs_ch_32 = generate_irs_channels(32, geo)

    print(f"\nMt={Mt}, Mr={Mr}, T={T}, P={P_dBm} dBm")
    print(f"IRS at ({pos_irs[0]:.0f}, {pos_irs[1]:.0f})")
    print(f"d_br={geo['d_br']:.1f}m, d_rt={geo['d_rt']:.1f}m")
    print(f"|alpha|^2 = {ch['alpha_sq']:.3e}")

    # ===== Run all scenarios =====
    # NLoS scenarios are slower (weak sensing → SDR/SCA takes longer)
    # so we run them with fewer SINR points
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

    # ===== Save and plot =====
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # Combined save
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

    # Plot 1: LoS comparison (no IRS vs IRS N=32 LoS)
    plot_irs_comparison(
        all_data.get("LoS, no IRS"),
        [all_data.get("LoS, IRS N=32")],
        labels=["IRS N=32 (LoS)"],
        save_path=os.path.join(out_dir, f'comparison_LoS_{timestamp}.png')
    )

    # Plot 2: NLoS comparison (all NLoS scenarios)
    nlos_labels = []
    nlos_data = []
    for lbl in ["NLoS, no IRS", "NLoS, IRS N=16", "NLoS, IRS N=32"]:
        if all_data.get(lbl) is not None:
            nlos_data.append(all_data[lbl])
            nlos_labels.append(lbl)

    savefig_path = os.path.join(out_dir, f'comparison_NLoS_{timestamp}.png')
    plot_irs_comparison(
        all_data.get("LoS, no IRS"),
        nlos_data,
        labels=nlos_labels,
        save_path=savefig_path
    )

    print("Done.")


if __name__ == '__main__':
    main()
