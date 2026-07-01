"""
场景扫描模块
============
对每个 SINR 阈值 γ₀ 求解优化问题，得到 CRB-Rate 曲线。

用法:
    from scenario import scan_scenario
    data = scan_scenario(label, use_irs, N_irs_scan, geo, ch, irs_ch, ...)
"""

import numpy as np
from config import (
    Mt, Mr, T, P, sigma2_c, sigma2_s,
    N_gamma, gamma_0_dB_min, gamma_0_dB_max,
)
from channels import compute_effective_a, compute_effective_h, irs_beam_align
from crb import compute_crb_irs
from rate import compute_rate_irs
from sca_solver import solve_p4_sca
from irs_solver import solve_irs_sdr, ao_optimize


def scan_scenario(label, use_irs, N_irs_scan, geo, ch, irs_ch,
                  direct_blocked=False, npts_override=None):
    """
    Run SINR sweep over gamma_0 for one scenario.
    Uses warm start: each gamma_0 point starts from the previous point's solution.
    """
    print(f"\n--- Scenario: {label} ---", flush=True)

    if direct_blocked and not use_irs:
        print(f"  (Direct path blocked, no IRS -- no sensing path)", flush=True)
        return None

    h = ch["h"]; a_dir = ch["a_dir"]; b = ch["b"]
    b_dot = ch["b_dot"]; alpha_sq = ch["alpha_sq"]

    npts = npts_override if npts_override else N_gamma
    gamma_0_dB_vals = np.linspace(gamma_0_dB_min, gamma_0_dB_max, npts)
    results = []

    prev_Rc = None
    prev_Rs = None

    for g0_dB in gamma_0_dB_vals:
        gamma_0 = 10**(g0_dB / 10)
        Rc = None

        if use_irs and direct_blocked:
            # NLoS + IRS: beam-align phases → single SCA
            v = irs_beam_align(irs_ch["h_r"], irs_ch["G"])
            a_eff = compute_effective_a(a_dir, irs_ch["G"], irs_ch["h_r"], v, direct_blocked=True)
            h_eff = compute_effective_h(h, irs_ch["G"], irs_ch["h_rc"], v)
            Rc, Rs, info = solve_p4_sca(gamma_0, h_eff, a_eff,
                sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq,
                Rc_init=prev_Rc, Rs_init=prev_Rs)

        elif use_irs and not direct_blocked:
            # LoS + IRS: full AO
            a_init = compute_effective_a(a_dir, irs_ch["G"], irs_ch["h_r"],
                                          np.ones(N_irs_scan, dtype=complex))
            h_init = compute_effective_h(h, irs_ch["G"], irs_ch["h_rc"],
                                          np.ones(N_irs_scan, dtype=complex))
            Rc, Rs, theta_opt, info = ao_optimize(
                gamma_0, a_init, h_init, irs_ch["G"], irs_ch["h_r"], irs_ch["h_rc"],
                b, b_dot, alpha_sq, sigma2_c, sigma2_s, P, Mt, Mr, T, N_irs_scan,
                direct_blocked=False, a_dir=a_dir)
            if Rc is not None:
                a_eff = compute_effective_a(a_dir, irs_ch["G"], irs_ch["h_r"], theta_opt)
                h_eff = compute_effective_h(h, irs_ch["G"], irs_ch["h_rc"], theta_opt)

        else:
            # LoS baseline: direct SCA
            Rc, Rs, info = solve_p4_sca(gamma_0, h, a_dir,
                sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq,
                Rc_init=prev_Rc, Rs_init=prev_Rs)
            if Rc is not None:
                a_eff = a_dir
                h_eff = h

        if Rc is None:
            results.append((gamma_0, None, None))
            prev_Rc, prev_Rs = None, None
            continue

        prev_Rc, prev_Rs = Rc.copy(), Rs.copy()

        rate, sinr = compute_rate_irs(Rc, Rs, h_eff, sigma2_c)
        crb = compute_crb_irs(Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)
        results.append((gamma_0, crb, rate))

    # Parse results
    valid = [(g, c, r) for g, c, r in results if c is not None]
    if len(valid) < 3:
        print(f"  {label}: too few feasible points ({len(valid)})", flush=True)
        return None

    g_arr, c_arr, r_arr = zip(*valid)
    data = {"gamma": np.array(g_arr), "crb": np.array(c_arr), "rate": np.array(r_arr)}
    print(f"  Feasible points: {len(valid)}/{npts}, "
          f"CRB range: [{c_arr[-1]:.3e}, {c_arr[0]:.3e}]", flush=True)
    return data
