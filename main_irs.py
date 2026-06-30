"""
IRS-assisted Bistatic ISAC — AO Framework Main Script
===================================================
基于 Paper 4 (CRB-Rate Tradeoff, TWC 2026) 扩展 IRS 辅助版本。
信号模型: Case 2 (叠加信号: 高斯信息 + 确定性感知信号)

AO 框架:
  Step 1: 固定 Theta, 优化 R_c, R_s (复用 case2_solver.py)
  Step 2: 固定 R_c, R_s, 优化 Theta (irs_solver.py SDR)

用法:
    python main_irs.py
"""

import os, time, numpy as np
import sys

from config import *            # All centralized parameters [ISAC-README §4]
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
# 主程序
# ========================================================================
def main():
    print("=" * 60)
    print("IRS-assisted Bistatic ISAC — CRB-Rate Tradeoff (Case 2)")
    print("=" * 60)

    np.random.seed(SEED)

    # ---- 1. Compute distances and angles ----
    d_bt = compute_distance(pos_bs, pos_target)       # BS -> Target [eq:6]
    d_tr = compute_distance(pos_target, pos_rx)        # Target -> RX
    d_br = compute_distance(pos_bs, pos_irs)           # BS -> IRS [new]
    d_rt = compute_distance(pos_irs, pos_target)       # IRS -> Target [new]

    # CU 位置: 距离 d_bc=1000m, 方向角 phi_cu=0.3 rad
    pos_cu = np.array([1000.0 * np.cos(phi_cu),
                       1000.0 * np.sin(phi_cu)])
    d_rc = compute_distance(pos_irs, pos_cu)           # IRS -> CU [new]
    d_bc = compute_distance(pos_bs, pos_cu)            # BS -> CU

    phi_br = compute_angle(pos_bs, pos_irs)            # BS -> IRS angle [new]
    phi_rt = compute_angle(pos_irs, pos_target)        # IRS -> Target angle [new]
    phi_rc = compute_angle(pos_irs, pos_cu)            # IRS -> CU angle [new]

    # ---- 2. Generate channels ----
    # Direct paths (Paper 4)
    # generate_rician_channel(Mt, phi_cu, Kc, d_bc, K0, alpha0, d0, seed=SEED_CHANNEL)
    h = generate_rician_channel(Mt, phi_cu, Kc, d_bc,
                                K0, alpha0, d0, SEED_CHANNEL)    # [eq:62]
    a = steering_vector(Mt, phi_target)                           # [eq:5a]
    b = steering_vector(Mr, theta_target)                         # [eq:5b]
    b_dot = steering_vector_derivative(Mr, theta_target)          # [eq:64]
    alpha_sq = compute_alpha_sq(d_bt, d_tr, 1.0,
                                K0, alpha0, d0, CAL_ALPHA)        # [eq:6]

    # IRS paths (new)
    G   = generate_irs_bs_channel(Mt, N_irs, d_br, phi_br,
                                  K0, alpha0, d0)                  # [new]
    h_r = generate_irs_target_channel(N_irs, d_rt, phi_rt,
                                       K0, alpha0, d0)             # [new]
    h_rc = generate_irs_cu_channel(N_irs, d_rc, phi_rc, Kc,
                                    K0, alpha0, d0, SEED_CHANNEL)  # [new]

    print(f"\nSystem: Mt={Mt}, Mr={Mr}, IRS N={N_irs}, T={T}, P={P_dBm} dBm")
    print(f"Geometry: BS(0,0), IRS({pos_irs[0]:.0f},{pos_irs[1]:.0f}), "
          f"Target(200,0), RX(400,0)")
    print(f"d_bt={d_bt:.1f}m, d_br={d_br:.1f}m, d_rt={d_rt:.1f}m")
    print(f"|alpha|^2 = {alpha_sq:.3e}")

    # ---- 3. SINR sweep ----
    gamma_0_dB_vals = np.linspace(gamma_0_dB_min, gamma_0_dB_max, N_gamma)

    # Without IRS (baseline — direct reuse of case2_solver)
    results_no_irs = []
    # With IRS
    results_irs = []

    print(f"\nSweeping {N_gamma} SINR thresholds...")
    print(f"{'gamma_0(dB)':>10} {'Base CRB':>14} {'IRS CRB':>14} "
          f"{'Base Rate':>14} {'IRS Rate':>14}")

    for g0_dB in gamma_0_dB_vals:
        gamma_0 = 10**(g0_dB / 10)

        # ---- Step A: Without IRS (baseline) ----
        Rc_base, Rs_base, info_base = solve_p4_sca(
            gamma_0, h, a, sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq
        )

        if Rc_base is None:
            results_no_irs.append((gamma_0, None, None))
            base_crb_str = '---'
            base_rate_str = '---'
        else:
            rate_b, sinr_b = compute_rate_irs(Rc_base, Rs_base, h, sigma2_c)
            crb_b = compute_crb_irs(theta_target, Rc_base, Rs_base, a,
                                     b, b_dot, alpha_sq, sigma2_s, T)
            results_no_irs.append((gamma_0, crb_b, rate_b))
            base_crb_str = f"{crb_b:.3e}"
            base_rate_str = f"{rate_b:.4f}"

        # ---- Step B: With IRS (AO) ----
        Rc_irs, Rs_irs, theta_opt, info_ao = ao_optimize(
            gamma_0, h, a, G, h_r, h_rc,
            b, b_dot, alpha_sq,
            sigma2_c, sigma2_s, P, Mt, Mr, T, N_irs
        )

        if Rc_irs is None:
            results_irs.append((gamma_0, None, None))
            irs_crb_str = '---'
            irs_rate_str = '---'
        else:
            a_eff = compute_effective_a(a, G, h_r, theta_opt)
            h_eff = compute_effective_h(h, G, h_rc, theta_opt)
            rate_i, sinr_i = compute_rate_irs(Rc_irs, Rs_irs, h_eff, sigma2_c)
            crb_i = compute_crb_irs(theta_target, Rc_irs, Rs_irs, a_eff,
                                     b, b_dot, alpha_sq, sigma2_s, T)
            results_irs.append((gamma_0, crb_i, rate_i))
            irs_crb_str = f"{crb_i:.3e}"
            irs_rate_str = f"{rate_i:.4f}"

        # Print status
        print(f"{g0_dB:>10.2f} {base_crb_str:>14} {irs_crb_str:>14} "
              f"{base_rate_str:>14} {irs_rate_str:>14}")

    # ---- 4. Save and plot ----
    _save_and_plot(results_no_irs, results_irs, gamma_0_dB_vals)
    print("\nDone.")


def ao_optimize(gamma_0, h, a, G, h_r, h_rc,
                b, b_dot, alpha_sq,
                sigma2_c, sigma2_s, P, Mt, Mr, T, N_irs):
    """
    AO: Alternating optimization for IRS-assisted ISAC.

    Iterates between:
      Step 1: Fix Theta, optimize R_c, R_s (case2_solver)
      Step 2: Fix R_c, R_s, optimize Theta (irs_solver)

    Returns:
        Rc_opt, Rs_opt: Optimized covariance matrices
        theta_opt: Optimized IRS phase shift vector
        info: dict with convergence info
    """
    # Initialize Theta = identity (zero phase shift)
    v = np.ones(N_irs, dtype=complex)
    a_eff = compute_effective_a(a, G, h_r, v)
    h_eff = compute_effective_h(h, G, h_rc, v)

    history = []

    for k in range(AO_MAX_ITER):
        # ---- Step 1: Fix Theta, optimize R_c, R_s ----
        Rc, Rs, info_sca = solve_p4_sca(
            gamma_0, h_eff, a_eff, sigma2_c, sigma2_s,
            P, Mt, Mr, b, b_dot, alpha_sq
        )

        if Rc is None:
            return None, None, None, {"status": f"SCA failed at AO iter {k}"}

        # ---- Step 2: Fix R_c, R_s, optimize Theta ----
        v_new, info_irs = solve_irs_sdr(
            Rc, Rs, a_eff, h_eff, G, h_r, h_rc,
            b, b_dot, alpha_sq,
            sigma2_c, sigma2_s, gamma_0,
            T, N_irs, Mt, trials=SDR_TRIALS
        )

        if v_new is None:
            # SDR failed — keep current Theta
            v_new = v

        # ---- Update effective channels ----
        a_eff_new = compute_effective_a(a, G, h_r, v_new)
        h_eff_new = compute_effective_h(h, G, h_rc, v_new)

        # ---- Compute CRB ----
        crb_old = compute_crb_irs(0, Rc, Rs, a_eff,
                                   b, b_dot, alpha_sq, sigma2_s, T)
        crb_new = compute_crb_irs(0, Rc, Rs, a_eff_new,
                                   b, b_dot, alpha_sq, sigma2_s, T)

        history.append({"iter": k, "crb": crb_new})

        # ---- Convergence check ----
        crb_change = abs(crb_new - crb_old) / (abs(crb_old) + 1e-15)
        v = v_new
        a_eff = a_eff_new
        h_eff = h_eff_new

        if crb_change < AO_TOL and k > 0:
            break

    return Rc, Rs, v, {"status": f"converged in {k+1} AO iters",
                       "history": history}


def _save_and_plot(results_no_irs, results_irs, gamma_0_dB_vals):
    """Save results and generate plots."""
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # Parse results
    def parse_results(results):
        valid = [(g, c, r) for g, c, r in results if c is not None]
        if len(valid) < 3:
            return None
        g_arr, c_arr, r_arr = zip(*valid)
        return {
            'gamma': np.array(g_arr),
            'crb': np.array(c_arr),
            'rate': np.array(r_arr)
        }

    data_base = parse_results(results_no_irs)
    data_irs  = parse_results(results_irs)

    # Save NPZ
    save_data = {}
    if data_base is not None:
        save_data['no_irs_gamma'] = data_base['gamma']
        save_data['no_irs_crb'] = data_base['crb']
        save_data['no_irs_rate'] = data_base['rate']
    if data_irs is not None:
        save_data['irs_gamma'] = data_irs['gamma']
        save_data['irs_crb'] = data_irs['crb']
        save_data['irs_rate'] = data_irs['rate']

    data_path = os.path.join(out_dir, f'irs_N{N_irs}_{timestamp}.npz')
    np.savez(data_path, **save_data)
    print(f"Data saved to: {data_path}")

    # Plot comparison
    fig_path = os.path.join(out_dir, f'irs_comparison_N{N_irs}_{timestamp}.png')
    plot_irs_comparison(
        data_base, [data_irs],
        labels=[f'IRS N={N_irs}'],
        save_path=fig_path
    )


if __name__ == '__main__':
    main()
