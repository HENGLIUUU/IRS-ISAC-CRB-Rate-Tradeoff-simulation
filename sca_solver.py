"""SCA solver for the Case-2 model in CRB-Rate Tradeoff for Bistatic ISAC."""

import numpy as np
import cvxpy as cp


def _bistatic_sca_weight(C, x_k):
    """
    Coefficient of a^T Rc a* in the affine SCA objective.

    It equals 1 - 1/(1+C*x_k)^2, obtained by combining the affine f1
    component in Eq. (58) and the Taylor lower bound in Eq. (59).
    """
    Cx = C * x_k
    return Cx * (2 + Cx) / (1 + Cx) ** 2


def _bistatic_nonlinear_gain(C, x):
    """Nonlinear communication contribution C*x²/(1+C*x) in Eq. (46)."""
    return C * x**2 / (1 + C * x)


def _bistatic_sca_lower_bound(C, x, x_k):
    """First-order global lower bound of the convex nonlinear gain."""
    return (
        _bistatic_nonlinear_gain(C, x_k)
        + _bistatic_sca_weight(C, x_k) * (x - x_k)
    )


def solve_p4_sca(gamma_0, h, a, sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq,
                 max_iter=50, tol=1e-4, Rc_init=None, Rs_init=None,
                 extra_noise_power=0.0,
                 irs_output_matrix=None,
                 irs_signal_power_budget=None):
    """
    Solve Problem (P4) using SCA  [Algorithm 1].

    Args:
        gamma_0: SINR threshold (linear scale, not dB)
        h: BS -> CU channel vector (Mt, 1)
        a: BS steering vector toward target (Mt, 1)
        sigma2_c: CU noise power (scalar)
        sigma2_s: Sensing receiver noise power (scalar)
        P: BS max transmit power (scalar, Watts)
        Mt, Mr: Number of antennas
        b: RX steering vector (Mr, 1)
        b_dot: Derivative of b (Mr, 1)
        alpha_sq: |alpha|^2 (scalar)
        extra_noise_power: additional receiver noise not contained in h,
            e.g. Active-IRS amplifier noise observed by the CU
        irs_output_matrix: optional K = G^H diag(|v|²) G for fixed-v
            Active-IRS output-power accounting
        irs_signal_power_budget: optional remaining IRS budget after its
            internally generated amplifier-noise output is subtracted
        max_iter: Maximum SCA iterations
        tol: Convergence tolerance

    Returns:
        Rc_opt: Optimal information covariance (Mt x Mt)
        Rs_opt: Optimal sensing covariance (Mt x Mt)
        info: dict with convergence info
    """
    h_flat = h.flatten()
    a_flat = a.flatten()
    b_flat = b.flatten()

    # Normalize the SINR constraint by the total receiver noise.
    if extra_noise_power < 0:
        raise ValueError("extra_noise_power must be non-negative.")

    # Active-IRS noise is independent of the BS signal. It changes the
    # receiver noise floor, but must not be absorbed into the channel h.
    total_c_noise = sigma2_c + extra_noise_power
    sigma_c_sqrt = np.sqrt(total_c_noise)
    h_tilde = h_flat / sigma_c_sqrt
    h_norm_sq = np.linalg.norm(h_tilde)**2

    max_SINR = P * np.linalg.norm(h_flat)**2 / total_c_noise
    if gamma_0 > max_SINR:
        return None, None, {"status": f"infeasible: gamma_0={10*np.log10(gamma_0):.1f}dB > max SINR ({10*np.log10(max_SINR):.1f}dB)"}

    C = alpha_sq * np.linalg.norm(b_flat)**2 / sigma2_s
    H_mat = np.outer(h_tilde, h_tilde.conj())
    # CRB-Rate Tradeoff for Bistatic ISAC uses a^T R a*. Since
    # trace(R a* a^T) = a^T R a*, the matrix is a* a^T.
    M_mat = np.outer(a_flat.conj(), a_flat)

    if Rc_init is not None and Rs_init is not None:
        Rc = Rc_init.copy()
        Rs = Rs_init.copy()
    else:
        # The MRT sensing beam is proportional to a*, because the
        # target response contains a^T x.
        sensing_beam = a_flat.conj() / np.linalg.norm(a_flat)
        h_a_corr_sq = np.abs(h_tilde.conj() @ sensing_beam)**2

        denom = h_norm_sq + gamma_0 * h_a_corr_sq
        numer = gamma_0 * (P * h_a_corr_sq + 1)
        Rc_power = min(numer / denom, P * 0.95)
        Rc = Rc_power * np.outer(h_flat, h_flat.conj()) / np.linalg.norm(h_flat)**2

        Rs_power = P - Rc_power
        if Rs_power <= 0:
            Rs = np.zeros((Mt, Mt), dtype=complex)
        else:
            Rs = Rs_power * np.outer(sensing_beam, sensing_beam.conj())

    history = []
    status = "max_iter_reached"

    for k in range(max_iter):
        Rc_old = Rc.copy()
        Rs_old = Rs.copy()

        x_k = float((a_flat.T @ Rc @ a_flat.conj()).real)
        Cx = C * x_k
        if Cx > 1e-15:
            h_prime = _bistatic_sca_weight(C, x_k)
        else:
            h_prime = 0.0

        # Convex subproblem at the current linearization point.
        Rc_var = cp.Variable((Mt, Mt), hermitian=True)
        Rs_var = cp.Variable((Mt, Mt), hermitian=True)

        # NLoS target channels can make every entry of M_mat extremely small.
        # A positive rescaling leaves the optimizer unchanged and prevents SCS
        # from treating the sensing objective as numerical zero.
        objective_scale = max(
            np.linalg.norm(M_mat, "fro") * max(1.0, abs(h_prime)),
            1e-30,
        )
        obj = cp.real(
            cp.trace(Rs_var @ M_mat) + h_prime * cp.trace(Rc_var @ M_mat)
        ) / objective_scale

        constraints = [
            Rc_var >> 0,
            Rs_var >> 0,
            cp.real(cp.trace(Rc_var) + cp.trace(Rs_var)) <= P,
            cp.real(cp.trace(Rc_var @ H_mat)) >= gamma_0 * (cp.real(cp.trace(Rs_var @ H_mat)) + 1.0),
        ]
        if irs_output_matrix is not None:
            if irs_signal_power_budget is None:
                raise ValueError(
                    "irs_signal_power_budget is required with irs_output_matrix."
                )
            if irs_signal_power_budget < 0:
                return None, None, {
                    "status": "infeasible: IRS amplifier noise exceeds budget"
                }
            constraints.append(
                cp.real(
                    cp.trace(
                        (Rc_var + Rs_var) @ irs_output_matrix
                    )
                ) <= irs_signal_power_budget
            )

        prob = cp.Problem(cp.Maximize(obj), constraints)
        try:
            prob.solve(
                solver=cp.SCS, verbose=False, eps=1e-5, max_iters=10000
            )
        except Exception as exc:
            return None, None, {"status": f"SCA solver error: {exc}"}

        if prob.status not in ("optimal", "optimal_inaccurate"):
            if k == 0:
                return None, None, {
                    "status": f"SCA solver returned {prob.status} at first iteration"
                }
            status = f"solver_{prob.status}"
            break

        Rc_new = Rc_var.value
        Rs_new = Rs_var.value

        Rc_change = np.linalg.norm(Rc_new - Rc_old) / (np.linalg.norm(Rc_old) + 1e-15)
        Rs_change = np.linalg.norm(Rs_new - Rs_old) / (np.linalg.norm(Rs_old) + 1e-15)
        obj_val = float((a_flat.T @ Rs_new @ a_flat.conj()).real
                        + h_prime * (a_flat.T @ Rc_new @ a_flat.conj()).real)
        x_new = float((a_flat.T @ Rc_new @ a_flat.conj()).real)
        y_new = float((a_flat.T @ Rs_new @ a_flat.conj()).real)
        x_old = float((a_flat.T @ Rc_old @ a_flat.conj()).real)
        y_old = float((a_flat.T @ Rs_old @ a_flat.conj()).real)
        exact_objective_old = y_old + _bistatic_nonlinear_gain(C, x_old)
        exact_objective_new = y_new + _bistatic_nonlinear_gain(C, x_new)
        lower_bound_new = y_new + _bistatic_sca_lower_bound(C, x_new, x_old)

        Rc, Rs = Rc_new, Rs_new
        history.append({
            "iter": k,
            "obj": obj_val,
            "Rc_change": Rc_change,
            "gamma_ran": Cx,
            "exact_objective_before": exact_objective_old,
            "exact_objective_after": exact_objective_new,
            "exact_objective_change": exact_objective_new - exact_objective_old,
            "surrogate_lower_bound_after": lower_bound_new,
            "lower_bound_gap": exact_objective_new - lower_bound_new,
        })

        if max(Rc_change, Rs_change) < tol:
            status = f"converged in {k+1} iters"
            break

    Rc = (Rc + Rc.conj().T) / 2
    Rs = (Rs + Rs.conj().T) / 2

    signal_power = float(np.real(np.vdot(h_flat, Rc @ h_flat)))
    sensing_interference = float(np.real(np.vdot(h_flat, Rs @ h_flat)))
    achieved_sinr = signal_power / max(
        sensing_interference + total_c_noise, 1e-30
    )
    used_power = float(np.real(np.trace(Rc + Rs)))
    min_eig_rc = float(np.min(np.linalg.eigvalsh(Rc)))
    min_eig_rs = float(np.min(np.linalg.eigvalsh(Rs)))
    feasibility_tol = 2e-3
    irs_signal_power = None
    if irs_output_matrix is not None:
        irs_signal_power = float(np.real(np.trace(
            (Rc + Rs) @ irs_output_matrix
        )))

    # Never expose a numerical iterate as a valid solution without checking
    # the original (un-normalized) constraints independently.
    infeasible = (
        achieved_sinr < gamma_0 * (1 - feasibility_tol)
        or used_power > P * (1 + feasibility_tol)
        or min_eig_rc < -feasibility_tol * max(P, 1.0)
        or min_eig_rs < -feasibility_tol * max(P, 1.0)
        or (
            irs_signal_power is not None
            and irs_signal_power
            > irs_signal_power_budget * (1 + feasibility_tol) + 1e-12
        )
    )
    if infeasible:
        return None, None, {
            "status": "SCA result failed independent feasibility check",
            "solver_status": prob.status,
            "sinr": achieved_sinr,
            "used_power": used_power,
            "min_eig_Rc": min_eig_rc,
            "min_eig_Rs": min_eig_rs,
            "irs_signal_power": irs_signal_power,
        }

    return Rc, Rs, {
        "status": status,
        "solver_status": prob.status,
        "history": history,
        "iters": k + 1,
        # These diagnostics let us distinguish a plotted point from a
        # genuinely feasible numerical solution.
        "sinr": achieved_sinr,
        "sinr_margin": achieved_sinr - gamma_0,
        "used_power": used_power,
        "power_margin": P - used_power,
        "min_eig_Rc": min_eig_rc,
        "min_eig_Rs": min_eig_rs,
        "communication_noise": total_c_noise,
        "irs_signal_power": irs_signal_power,
    }
