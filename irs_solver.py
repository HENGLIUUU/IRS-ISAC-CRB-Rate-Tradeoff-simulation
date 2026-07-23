"""
IRS reflection-coefficient optimization for the simplified bistatic model.

This module keeps the two conventions used by the source models:

    communication: y_c = h_eff^H x,
    bistatic sensing: y_s = alpha b a_eff^T x + n.

Thus

    h_eff = h_direct + G^H diag(conj(v)) h_cu,
    a_eff = a_direct + G^T diag(v) h_target.

For the SDP we optimize u = conj(v). The communication channel h_eff and
the conjugated target vector q_eff = conj(a_eff) are affine in u.

    q_eff = conj(a_direct) + conj(G^T diag(h_target)) u,
    h_eff = h_direct + G^H diag(h_cu) u.

Important modelling boundary
----------------------------
The CRB follows CRB-Rate Tradeoff for Bistatic ISAC with an equivalent
target channel.
The Active-IRS noise is included in the CU SINR and IRS output-power
constraint. The first-pass noise reflected by the target is separately
quantified and verified to be negligible under the configured geometry, so
the sensing CRB retains the source model's white-noise assumption. This is
a transparent one-pass extension, not a reproduction of the double-pass FIM
in Cramér-Rao Bound Optimization for Active RIS-Empowered ISAC Systems.
"""

import numpy as np
import cvxpy as cp

from channels import (
    compute_active_irs_noise_power,
    compute_effective_a,
    compute_effective_h,
    compute_safe_uniform_active_gain,
    irs_beam_align,
)
from config import AO_MAX_ITER, AO_TOL, SDR_TRIALS, A_MAX, P_RIS, SIGMA2_RIS
from crb import compute_crb_irs
from sca_solver import solve_p4_sca


def _build_effective_channel_map(G, h_irs):
    """
    Return B such that G^H diag(conj(v)) h_irs = B u, u=conj(v).
    """
    h_col = np.asarray(h_irs).reshape(-1)
    return G.conj().T @ np.diag(h_col)


def _build_target_steering_map(G, h_target):
    """Return B such that a_eff = a_direct + B v for bistatic sensing."""
    h_col = np.asarray(h_target).reshape(-1)
    return G.T @ np.diag(h_col)


def _augment_quadratic(M, linear, constant):
    """
    Build Q with [u;1]^H Q [u;1] = u^H M u + 2 Re(u^H linear)+constant.
    """
    n = M.shape[0]
    Q = np.zeros((n + 1, n + 1), dtype=complex)
    Q[:n, :n] = (M + M.conj().T) / 2
    Q[:n, n] = linear
    Q[n, :n] = linear.conj()
    Q[n, n] = float(np.real(constant))
    return Q


def _active_irs_power_matrix(G, Rc, Rs, sigma2_irs):
    """
    Return diagonal D for P_IRS = u^H D u = v^H D v.

    The n-th diagonal entry is the incident signal power at IRS element n
    plus that element's internal amplifier-noise power.
    """
    Rx = Rc + Rs
    incident_covariance = G @ Rx @ G.conj().T
    per_element_input = np.real(np.diag(incident_covariance)) + sigma2_irs
    return np.diag(np.maximum(per_element_input, 0.0))


def _fixed_v_sca_power_constraint(G, v, sigma2_irs, power_budget):
    """
    Return K and residual budget for fixed-v SCA:

        tr((Rc+Rs) K) <= P_RIS - sigma_I² ||v||²,
        K = G^H diag(|v|²) G.
    """
    v_abs_sq = np.abs(np.asarray(v).reshape(-1)) ** 2
    K = G.conj().T @ np.diag(v_abs_sq) @ G
    residual = float(power_budget - sigma2_irs * np.sum(v_abs_sq))
    return (K + K.conj().T) / 2, residual


def _evaluate_candidate(
    u,
    Rc,
    Rs,
    a_direct,
    h_direct,
    target_map,
    C_map,
    h_rc,
    gamma_0,
    sigma2_c,
    sigma2_irs,
    power_matrix,
    active,
    b,
    b_dot,
    alpha_sq,
    sigma2_s,
    T,
    feasibility_tol=2e-3,
):
    """
    Check a recovered SDP candidate against the original constraints.

    SDP relaxation feasibility does not guarantee that a randomized rank-one
    vector is feasible. Every candidate is therefore checked again before its
    CRB is allowed to compete for "best".
    """
    u = np.asarray(u).reshape(-1)
    v = u.conj()

    a_eff = np.asarray(a_direct).reshape(-1) + target_map @ v
    h_eff = np.asarray(h_direct).reshape(-1) + C_map @ u

    signal = float(np.real(h_eff.conj() @ Rc @ h_eff))
    sensing_interference = float(np.real(h_eff.conj() @ Rs @ h_eff))
    irs_noise = (
        compute_active_irs_noise_power(h_rc, v, sigma2_irs)
        if active else 0.0
    )
    denominator = sensing_interference + sigma2_c + irs_noise
    sinr = signal / max(denominator, 1e-30)

    irs_power = float(np.real(u.conj() @ power_matrix @ u))
    amplitude_ok = (
        np.max(np.abs(v)) <= A_MAX * (1 + feasibility_tol)
        if active
        else np.max(np.abs(np.abs(v) - 1.0)) <= feasibility_tol
    )
    power_ok = (not active) or irs_power <= P_RIS * (1 + feasibility_tol)
    sinr_ok = sinr >= gamma_0 * (1 - feasibility_tol)

    if not (amplitude_ok and power_ok and sinr_ok):
        return None

    crb = compute_crb_irs(
        Rc, Rs, a_eff.reshape(-1, 1), b, b_dot,
        alpha_sq, sigma2_s, T,
    )
    return {
        "u": u,
        "v": v,
        "crb": float(crb),
        "sinr": float(sinr),
        "irs_power": irs_power,
    }


def _candidate_vectors(V_opt, active, trials, rng):
    """
    Yield deterministic and randomized u candidates from augmented SDP V.

    Sampling the full augmented matrix is essential because its final element
    carries the linear terms. After sampling y=[u_raw;t], divide by t so the
    augmented coordinate becomes one.
    """
    V_hermitian = (V_opt + V_opt.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(V_hermitian)
    eigvals = np.maximum(eigvals, 0.0)

    # Principal-eigenvector candidate is deterministic and often strong.
    principal = eigvecs[:, -1] * np.sqrt(eigvals[-1])
    raw_candidates = [principal]

    sqrt_V = eigvecs @ np.diag(np.sqrt(eigvals))
    for _ in range(trials):
        xi = (
            rng.standard_normal(V_opt.shape[0])
            + 1j * rng.standard_normal(V_opt.shape[0])
        ) / np.sqrt(2)
        raw_candidates.append(sqrt_V @ xi)

    for y in raw_candidates:
        if abs(y[-1]) < 1e-10:
            continue
        u = y[:-1] / y[-1]
        if active:
            magnitude = np.abs(u)
            u = np.where(
                magnitude > A_MAX,
                A_MAX * u / (magnitude + 1e-15),
                u,
            )
        else:
            u = u / (np.abs(u) + 1e-15)
        yield u


def solve_irs_sdr(
    Rc,
    Rs,
    a_direct,
    h_direct,
    G,
    h_r,
    h_rc,
    b,
    b_dot,
    alpha_sq,
    sigma2_c,
    sigma2_s,
    gamma_0,
    T,
    N_irs,
    Mt,
    trials=100,
    active=True,
    v_reference=None,
    random_seed=0,
    sigma2_irs=SIGMA2_RIS,
):
    """
    Optimize IRS coefficients for fixed Rc and Rs using an SDR surrogate.

    The exact bistatic CRB is non-quadratic in a_eff because its Gaussian
    information-signal weight depends on a_eff^T Rc a_eff*. We freeze that
    weight at v_reference, producing one documented quadratic surrogate.
    AO updates the reference and therefore refreshes the surrogate.
    """
    del Mt  # dimensions are inferred and validated from the arrays
    if sigma2_irs < 0:
        raise ValueError("sigma2_irs must be non-negative.")

    a_direct = np.asarray(a_direct).reshape(-1)
    h_direct = np.asarray(h_direct).reshape(-1)
    h_r_col = np.asarray(h_r).reshape(-1, 1)
    h_rc_col = np.asarray(h_rc).reshape(-1, 1)
    if G.shape[0] != N_irs:
        raise ValueError("N_irs does not match the number of rows in G.")

    target_map = _build_target_steering_map(G, h_r_col)
    C_map = _build_effective_channel_map(G, h_rc_col)

    if v_reference is None:
        v_reference = np.ones(N_irs, dtype=complex)
    u_reference = np.asarray(v_reference).reshape(-1).conj()
    a_reference = a_direct + target_map @ np.asarray(v_reference).reshape(-1)

    aT_Rc_astar = float(np.real(a_reference @ Rc @ a_reference.conj()))
    gamma_ran = (
        alpha_sq * aT_Rc_astar * np.linalg.norm(b) ** 2 / sigma2_s
    )
    weight = gamma_ran / (1 + gamma_ran) if gamma_ran > 0 else 0.0

    # Frozen-weight sensing surrogate:
    # a_eff^T (Rs + weight Rc) a_eff*. With q_eff=conj(a_eff),
    # this is q_eff^H (Rs + weight Rc) q_eff and is quadratic in u.
    R_total = Rs + weight * Rc
    q_direct = a_direct.conj()
    Q_map = target_map.conj()
    M_obj = Q_map.conj().T @ R_total @ Q_map
    l_obj = Q_map.conj().T @ R_total @ q_direct
    c_obj = np.real(q_direct.conj() @ R_total @ q_direct)
    Q_obj = _augment_quadratic(M_obj, l_obj, c_obj)

    # Communication constraint:
    # h_eff^H(Rc-gamma Rs)h_eff
    # - gamma * active-IRS-noise >= gamma*sigma_c².
    R_sinr = Rc - gamma_0 * Rs
    M_sinr = C_map.conj().T @ R_sinr @ C_map
    l_sinr = C_map.conj().T @ R_sinr @ h_direct
    c_sinr = (
        np.real(h_direct.conj() @ R_sinr @ h_direct)
        - gamma_0 * sigma2_c
    )
    if active:
        noise_diagonal = sigma2_irs * np.abs(h_rc_col.reshape(-1)) ** 2
        M_sinr = M_sinr - gamma_0 * np.diag(noise_diagonal)
    Q_sinr = _augment_quadratic(M_sinr, l_sinr, c_sinr)

    power_matrix = _active_irs_power_matrix(
        G, Rc, Rs, sigma2_irs if active else 0.0
    )

    # Hermitian is not cosmetic: a generic complex variable has a complex
    # diagonal, for which CVXPY correctly rejects real inequalities.
    V = cp.Variable((N_irs + 1, N_irs + 1), hermitian=True)
    constraints = [V >> 0, V[N_irs, N_irs] == 1]
    diagonal = cp.real(cp.diag(V)[:N_irs])
    if active:
        constraints.extend([
            diagonal <= A_MAX**2,
            cp.real(cp.trace(power_matrix @ V[:N_irs, :N_irs])) <= P_RIS,
        ])
    else:
        constraints.append(diagonal == 1)
    constraints.append(cp.real(cp.trace(Q_sinr @ V)) >= 0)

    problem = cp.Problem(
        cp.Maximize(cp.real(cp.trace(Q_obj @ V))),
        constraints,
    )
    try:
        problem.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-4,
            max_iters=10000,
        )
    except Exception as exc:
        return None, {"status": f"SDP solver error: {exc}"}

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None, {"status": f"SDP infeasible: {problem.status}"}

    rng = np.random.default_rng(random_seed)
    best = None
    feasible_count = 0
    for u in _candidate_vectors(V.value, active, trials, rng):
        candidate = _evaluate_candidate(
            u, Rc, Rs, a_direct, h_direct, target_map, C_map, h_rc_col,
            gamma_0, sigma2_c, sigma2_irs, power_matrix, active,
            b, b_dot, alpha_sq, sigma2_s, T,
        )
        if candidate is None:
            continue
        feasible_count += 1
        if best is None or candidate["crb"] < best["crb"]:
            best = candidate

    if best is None:
        return None, {
            "status": "SDP solved but rank-one recovery found no feasible vector",
            "sdp_status": problem.status,
        }

    return best["v"], {
        "status": problem.status,
        "crb": best["crb"],
        "sinr": best["sinr"],
        "irs_power": best["irs_power"],
        "sdp_objective": float(problem.value),
        "feasible_candidates": feasible_count,
        "trials_used": trials,
    }


def ao_optimize(
    gamma_0,
    a_direct,
    h_direct,
    G,
    h_r,
    h_rc,
    b,
    b_dot,
    alpha_sq,
    sigma2_c,
    sigma2_s,
    P,
    Mt,
    Mr,
    T,
    N_irs,
    direct_blocked=False,
    cu_direct_blocked=False,
    active=True,
):
    """
    Alternate between BS covariance optimization and IRS optimization.

    Direct channels remain immutable. Every effective channel is recomputed
    from the latest v, preventing reflected paths from accumulating across AO
    iterations.
    """
    a_base = (
        np.zeros_like(np.asarray(a_direct).reshape(-1, 1))
        if direct_blocked else np.asarray(a_direct).reshape(-1, 1)
    )
    h_base = (
        np.zeros_like(np.asarray(h_direct).reshape(-1, 1))
        if cu_direct_blocked else np.asarray(h_direct).reshape(-1, 1)
    )

    # Start AO from the closed-form target-alignment baseline. Starting from
    # all-one phases can trap the frozen-weight SDR sequence at a solution
    # that is much worse than the inexpensive baseline it is meant to improve.
    v = irs_beam_align(h_r, G)
    if active:
        # The initial point must already satisfy the IRS power constraint,
        # because a later non-improving SDR step may be rejected and AO will
        # legitimately return this vector.
        v *= compute_safe_uniform_active_gain(
            G, P, P_RIS, SIGMA2_RIS, A_MAX
        )

    history = []
    Rc = Rs = None
    status = "max_iter_reached"

    for iteration in range(AO_MAX_ITER):
        a_eff = compute_effective_a(a_base, G, h_r, v)
        h_eff = compute_effective_h(h_base, G, h_rc, v)
        irs_noise = (
            compute_active_irs_noise_power(h_rc, v, SIGMA2_RIS)
            if active else 0.0
        )
        if active:
            irs_output_matrix, irs_signal_budget = (
                _fixed_v_sca_power_constraint(G, v, SIGMA2_RIS, P_RIS)
            )
        else:
            irs_output_matrix = irs_signal_budget = None

        Rc, Rs, sca_info = solve_p4_sca(
            gamma_0, h_eff, a_eff,
            sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq,
            extra_noise_power=irs_noise,
            irs_output_matrix=irs_output_matrix,
            irs_signal_power_budget=irs_signal_budget,
        )
        if Rc is None:
            return None, None, None, {
                "status": f"SCA failed at AO iteration {iteration}",
                "sca": sca_info,
            }

        old_crb = compute_crb_irs(
            Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T
        )
        v_new, irs_info = solve_irs_sdr(
            Rc, Rs, a_base, h_base, G, h_r, h_rc,
            b, b_dot, alpha_sq, sigma2_c, sigma2_s,
            gamma_0, T, N_irs, Mt,
            trials=SDR_TRIALS,
            active=active,
            v_reference=v,
            random_seed=iteration,
        )
        if v_new is None:
            status = f"IRS recovery stopped at iteration {iteration}"
            break

        a_new = compute_effective_a(a_base, G, h_r, v_new)
        new_crb = compute_crb_irs(
            Rc, Rs, a_new, b, b_dot, alpha_sq, sigma2_s, T
        )
        relative_change = abs(new_crb - old_crb) / (abs(old_crb) + 1e-30)
        accepted = new_crb <= old_crb * (1 + 1e-8)
        history.append({
            "iter": iteration,
            "crb_before_irs": float(old_crb),
            "crb_after_irs": float(new_crb),
            "relative_change": float(relative_change),
            "irs_status": irs_info["status"],
            "accepted": accepted,
        })

        # SDR optimizes a frozen-weight quadratic surrogate, not the exact
        # CRB. Randomization can therefore produce a feasible vector whose
        # exact CRB is worse. AO should never accept such a step.
        if not accepted:
            status = f"stopped at non-improving IRS step {iteration}"
            break

        v = v_new

        if relative_change < AO_TOL and iteration > 0:
            status = f"converged in {iteration + 1} AO iterations"
            break

    # Rc and Rs above were optimized for the previous v. Solve once more so
    # the returned triplet (Rc, Rs, v) is internally consistent.
    a_final = compute_effective_a(a_base, G, h_r, v)
    h_final = compute_effective_h(h_base, G, h_rc, v)
    final_irs_noise = (
        compute_active_irs_noise_power(h_rc, v, SIGMA2_RIS)
        if active else 0.0
    )
    if active:
        final_output_matrix, final_signal_budget = (
            _fixed_v_sca_power_constraint(
                G, v, SIGMA2_RIS, P_RIS
            )
        )
    else:
        final_output_matrix = final_signal_budget = None
    Rc, Rs, final_sca = solve_p4_sca(
        gamma_0, h_final, a_final,
        sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq,
        extra_noise_power=final_irs_noise,
        irs_output_matrix=final_output_matrix,
        irs_signal_power_budget=final_signal_budget,
    )
    if Rc is None:
        return None, None, None, {
            "status": "Final consistency SCA failed",
            "history": history,
            "sca": final_sca,
        }

    final_signal = float(
        np.real(h_final.conj().T @ Rc @ h_final).item()
    )
    final_interference = float(
        np.real(h_final.conj().T @ Rs @ h_final).item()
    )
    final_sinr = final_signal / max(
        final_interference + sigma2_c + final_irs_noise, 1e-30
    )
    final_power_matrix = _active_irs_power_matrix(
        G, Rc, Rs, SIGMA2_RIS if active else 0.0
    )
    final_irs_power = float(np.real(v.conj() @ final_power_matrix @ v))
    feasibility_tol = 2e-3
    if (
        final_sinr < gamma_0 * (1 - feasibility_tol)
        or (active and final_irs_power > P_RIS * (1 + feasibility_tol))
        or (active and np.max(np.abs(v)) > A_MAX * (1 + feasibility_tol))
    ):
        return None, None, None, {
            "status": "Final AO solution failed independent feasibility check",
            "history": history,
            "sinr": final_sinr,
            "irs_power": final_irs_power,
        }

    return Rc, Rs, v, {
        "status": status,
        "history": history,
        "final_sca": final_sca["status"],
        "sinr": final_sinr,
        "irs_power": final_irs_power,
        "max_amplitude": float(np.max(np.abs(v))),
        "final_crb": float(compute_crb_irs(
            Rc, Rs, a_final, b, b_dot, alpha_sq, sigma2_s, T
        )),
    }
