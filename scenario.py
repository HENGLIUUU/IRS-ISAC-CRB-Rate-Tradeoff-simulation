"""SINR sweep orchestration and result diagnostics."""

import numpy as np
from config import (
    A_MAX, P_RIS, SIGMA2_RIS,
    Mt, Mr, T, P, sigma2_c, sigma2_s,
    N_gamma, gamma_0_dB_min, gamma_0_dB_max,
)
from channels import (
    compute_effective_a,
    compute_effective_h,
    compute_active_irs_noise_power,
    compute_active_irs_output_power,
    compute_forwarded_irs_sensing_noise_power,
    compute_safe_uniform_active_gain,
    irs_beam_align,
)
from crb import compute_crb_irs
from rate import compute_rate_irs
from sca_solver import solve_p4_sca
from irs_solver import solve_irs_sdr, ao_optimize


def scan_scenario(label, use_irs, N_irs_scan, geo, ch, irs_ch,
                  active=False,
                  direct_blocked=False, npts_override=None,
                  irs_strategy="alignment",
                  bs_power_override=None):
    """
    Run SINR sweep over gamma_0 for one scenario.
    Uses warm start: each gamma_0 point starts from the previous point's solution.

    irs_strategy="alignment" runs the inexpensive phase-alignment baseline.
    irs_strategy="ao" runs the much heavier alternating SCA/SDR optimization.
    """
    print(f"\n--- Scenario: {label} ---", flush=True)
    bs_power = P if bs_power_override is None else float(bs_power_override)
    if bs_power <= 0:
        raise ValueError("BS power must be positive.")

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
        # Passive IRS creates no amplifier noise. Active scenarios overwrite
        # this value after their reflection vector v is known.
        irs_noise_power = 0.0
        irs_output_power = 0.0
        forwarded_sensing_noise = 0.0
        irs_gain = 1.0 if use_irs else 0.0
        v_used = None

        if use_irs and direct_blocked and irs_strategy == "ao":
            Rc, Rs, theta_opt, info = ao_optimize(
                gamma_0, a_dir, h, irs_ch["G"], irs_ch["h_r"], irs_ch["h_rc"],
                b, b_dot, alpha_sq, sigma2_c, sigma2_s,
                bs_power, Mt, Mr, T, N_irs_scan,
                direct_blocked=True, cu_direct_blocked=False, active=active,
            )
            if Rc is not None:
                v_used = theta_opt
                a_eff = compute_effective_a(
                    a_dir, irs_ch["G"], irs_ch["h_r"], theta_opt,
                    direct_blocked=True,
                )
                h_eff = compute_effective_h(
                    h, irs_ch["G"], irs_ch["h_rc"], theta_opt
                )
                if active:
                    irs_gain = float(np.max(np.abs(theta_opt)))
                    irs_noise_power = compute_active_irs_noise_power(
                        irs_ch["h_rc"], theta_opt, SIGMA2_RIS
                    )

        elif use_irs and direct_blocked:
            # NLoS + IRS: beam-align phases → single SCA
            v = irs_beam_align(irs_ch["h_r"], irs_ch["G"])
            if active:
                # Fast uniform-gain baseline: obey both the per-element
                # amplitude limit and a guaranteed total output-power bound.
                safe_gain = compute_safe_uniform_active_gain(
                    irs_ch["G"], bs_power, P_RIS, SIGMA2_RIS, A_MAX
                )
                v = v * safe_gain
                irs_gain = safe_gain
            v_used = v
            a_eff = compute_effective_a(a_dir, irs_ch["G"], irs_ch["h_r"], v, direct_blocked=True)
            h_eff = compute_effective_h(h, irs_ch["G"], irs_ch["h_rc"], v)
            if active:
                irs_noise_power = compute_active_irs_noise_power(
                    irs_ch["h_rc"], v, SIGMA2_RIS
                )
            Rc, Rs, info = solve_p4_sca(gamma_0, h_eff, a_eff,
                sigma2_c, sigma2_s, bs_power, Mt, Mr, b, b_dot, alpha_sq,
                Rc_init=prev_Rc, Rs_init=prev_Rs,
                extra_noise_power=irs_noise_power)

        elif use_irs and not direct_blocked:
            # LoS + IRS: full AO
            Rc, Rs, theta_opt, info = ao_optimize(
                gamma_0, a_dir, h, irs_ch["G"], irs_ch["h_r"], irs_ch["h_rc"],
                b, b_dot, alpha_sq, sigma2_c, sigma2_s, bs_power, Mt, Mr, T, N_irs_scan,
                direct_blocked=False, cu_direct_blocked=False, active=active)
            if Rc is not None:
                v_used = theta_opt
                a_eff = compute_effective_a(a_dir, irs_ch["G"], irs_ch["h_r"], theta_opt)
                h_eff = compute_effective_h(h, irs_ch["G"], irs_ch["h_rc"], theta_opt)
                if active:
                    irs_gain = float(np.max(np.abs(theta_opt)))
                    irs_noise_power = compute_active_irs_noise_power(
                        irs_ch["h_rc"], theta_opt, SIGMA2_RIS
                    )

        else:
            # LoS baseline: direct SCA
            Rc, Rs, info = solve_p4_sca(gamma_0, h, a_dir,
                sigma2_c, sigma2_s, bs_power, Mt, Mr, b, b_dot, alpha_sq,
                Rc_init=prev_Rc, Rs_init=prev_Rs)
            if Rc is not None:
                a_eff = a_dir
                h_eff = h

        if Rc is None:
            results.append((
                gamma_0, None, None, None, None, None, None, None
            ))
            prev_Rc, prev_Rs = None, None
            continue

        prev_Rc, prev_Rs = Rc.copy(), Rs.copy()

        rate, sinr = compute_rate_irs(
            Rc, Rs, h_eff, sigma2_c,
            irs_noise_power=irs_noise_power,
        )
        crb = compute_crb_irs(Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)
        if active:
            irs_output_power = compute_active_irs_output_power(
                irs_ch["G"], Rc, Rs, v_used, SIGMA2_RIS
            )
            forwarded_sensing_noise = compute_forwarded_irs_sensing_noise_power(
                irs_ch["h_r"], v_used, b, alpha_sq, SIGMA2_RIS
            )
        results.append((
            gamma_0, crb, rate, sinr, irs_noise_power,
            irs_output_power, irs_gain, forwarded_sensing_noise,
        ))

    # Parse results
    valid = [
        (g, c, r, s, n, p, gain, sensing_n)
        for g, c, r, s, n, p, gain, sensing_n in results if c is not None
    ]
    if len(valid) < 3:
        print(f"  {label}: too few feasible points ({len(valid)})", flush=True)
        return None

    (
        g_arr, c_arr, r_arr, s_arr, n_arr, p_arr, gain_arr,
        sensing_n_arr,
    ) = zip(*valid)
    data = {
        "gamma": np.array(g_arr),
        "crb": np.array(c_arr),
        "rate": np.array(r_arr),
        "sinr": np.array(s_arr),
        "irs_noise": np.array(n_arr),
        "irs_output_power": np.array(p_arr),
        "irs_gain": np.array(gain_arr),
        "forwarded_sensing_noise": np.array(sensing_n_arr),
    }
    print(f"  Feasible points: {len(valid)}/{npts}, "
          f"CRB range: [{c_arr[-1]:.3e}, {c_arr[0]:.3e}], "
          f"min SINR margin: {10*np.log10(np.min(np.array(s_arr)/np.array(g_arr))):+.2f} dB",
          flush=True)
    if active:
        print(
            f"  Active diagnostics: max gain={np.max(gain_arr):.3g}, "
            f"IRS output={10*np.log10(np.min(p_arr))+30:.2f} to "
            f"{10*np.log10(np.max(p_arr))+30:.2f} dBm, "
            f"max CU noise/thermal={np.max(n_arr)/sigma2_c:.3e}, "
            f"max forwarded sensing noise/thermal="
            f"{np.max(sensing_n_arr)/sigma2_s:.3e}",
            flush=True,
        )
    return data
