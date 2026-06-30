"""
IRS 相移优化求解器 — SDR (Semidefinite Relaxation)
================================================
对应设计文档 §3.3。

固定 R_c, R_s 后，优化 Θ = diag(v) 以最小化 CRB。

求解步骤:
  1. 将目标函数和 SINR 约束写成 v 的二次型
  2. 松弛 rank-1 约束 → SDP (V ⪰ 0, diag(V) = 1)
  3. CVXPY 求解
  4. 随机化恢复 rank-1 解

用法:
    from irs_solver import solve_irs_sdr
    v_opt, info = solve_irs_sdr(Rc, Rs, ...)
    theta_opt = np.diag(v_opt)
"""

import numpy as np
import cvxpy as cp
from channels import compute_effective_a, compute_effective_h


def _build_a_eff_linear_map(G, h_r, N_irs, Mt):
    """
    Build linear map A such that a_eff(v) = a + A @ v.

    a_eff(v) = a + G^T @ (h_r^T * v)
    A = G^T @ diag(h_r.flatten()) 属于 C^{Mt x N_irs}

    Returns:
        A: Matrix mapping v -> IRS contribution to a_eff
        a_const: Direct-path component (Mt,)
    """
    h_r_flat = h_r.flatten()  # (N_irs,)
    A = G.T @ np.diag(h_r_flat)  # (Mt x N_irs)
    return A


def _build_h_eff_linear_map(G, h_rc, N_irs, Mt):
    """
    Build linear map C such that h_eff(v) = h + C @ v.

    Returns:
        C: Matrix mapping v -> IRS contribution to h_eff
        h_const: Direct-path component (Mt,)
    """
    h_rc_flat = h_rc.flatten()  # (N_irs,)
    C = G.T @ np.diag(h_rc_flat)  # (Mt x N_irs)
    return C


def solve_irs_sdr(Rc, Rs, a, h, G, h_r, h_rc,
                  b, b_dot, alpha_sq,
                  sigma2_c, sigma2_s, gamma_0,
                  T, N_irs, Mt, trials=100):
    """
    Solve IRS phase shift optimization via SDR.

    Fixed R_c, R_s -> optimize Theta to minimize CRB.
    Formulated as SDP + randomization.

    Args:
        Rc: Information covariance (Mt x Mt) -- fixed from AO Step 1
        Rs: Sensing covariance (Mt x Mt) -- fixed from AO Step 1
        a: Direct steering vector (Mt x 1)
        h: Direct CU channel (Mt x 1)
        G: BS -> IRS channel (N_irs x Mt)
        h_r: IRS -> Target channel (1 x N_irs)
        h_rc: IRS -> CU channel (1 x N_irs)
        b: RX steering vector (Mr x 1)
        b_dot: RX steering derivative (Mr x 1)
        alpha_sq: |alpha|^2 target coefficient
        sigma2_c, sigma2_s: Noise powers
        gamma_0: SINR threshold (linear)
        T: Symbols
        N_irs: Number of IRS elements
        Mt: BS antennas
        trials: Number of randomization trials

    Returns:
        v_opt: Optimal phase shift vector (N_irs,), |v_opt[n]| = 1
        info: dict with convergence info
    """
    a_flat = a.flatten()
    h_flat = h.flatten()
    b_flat = b.flatten()

    norm_b_sq = np.linalg.norm(b_flat) ** 2
    norm_bdot_sq = np.linalg.norm(b_dot) ** 2

    # ---- Build linear maps ----
    A_mat = _build_a_eff_linear_map(G, h_r, N_irs, Mt)  # Mt x N_irs
    C_mat = _build_h_eff_linear_map(G, h_rc, N_irs, Mt)  # Mt x N_irs

    # ---- Precompute quadratic form matrices ----
    # a_eff^H R a_eff = (a + Av)^H R (a + Av)
    # = a^H R a + 2 Re(a^H R A v) + v^H A^H R A v

    # For Rc:
    aH_Rc_a = float((a_flat.conj() @ Rc @ a_flat).real)
    # For Rs:
    aH_Rs_a = float((a_flat.conj() @ Rs @ a_flat).real)

    # Quadratic part: M_Rc = A^H Rc A  (N_irs x N_irs)
    M_Rc = A_mat.conj().T @ Rc @ A_mat
    M_Rs = A_mat.conj().T @ Rs @ A_mat
    M_hRc = C_mat.conj().T @ Rc @ C_mat
    M_hRs = C_mat.conj().T @ Rs @ C_mat

    # Linear part: l_Rc = A^H Rc a  (N_irs,)
    l_Rc = A_mat.conj().T @ Rc @ a_flat   # N_irs,
    l_Rs = A_mat.conj().T @ Rs @ a_flat

    # For SINR: h_eff^H R_c h_eff, h_eff^H R_s h_eff
    l_hRc = C_mat.conj().T @ Rc @ h_flat
    l_hRs = C_mat.conj().T @ Rs @ h_flat

    # ---- Compute fixed weight w = gamma_ran / (1 + gamma_ran) for the objective ----
    # Use current Rc's a_eff^H R_c a_eff to compute gamma_ran
    # This is approximated: we fix w during this SDR step
    # Use v=0 (no IRS) for initial gamma_ran estimate
    gamma_ran = alpha_sq * aH_Rc_a * norm_b_sq / sigma2_s
    w = gamma_ran / (1 + gamma_ran) if gamma_ran > 0 else 0

    # ---- Simplified: fixed weight w, maximize F = a_eff^H (Rs + w * Rc) a_eff ----
    # This is equivalent to minimizing CRB when w is fixed
    R_tot = Rs + w * Rc
    M_tot = A_mat.conj().T @ R_tot @ A_mat  # N_irs x N_irs
    l_tot = A_mat.conj().T @ R_tot @ a_flat  # N_irs,

    const_obj = float((a_flat.conj() @ R_tot @ a_flat).real)  # constant term

    # ---- SINR constraint (cross-multiplied) ----
    # h_eff^H R_c h_eff >= gamma_0 * (h_eff^H R_s h_eff + sigma2_c)
    # -> v^H (M_hRc - gamma_0 M_hRs) v + 2 Re(l_h_combined^H v) + const_sinr >= 0
    M_sinr = M_hRc - gamma_0 * M_hRs  # N_irs x N_irs
    l_sinr = l_hRc - gamma_0 * l_hRs  # N_irs,
    const_sinr = (float((h_flat.conj() @ Rc @ h_flat).real)
                  - gamma_0 * (float((h_flat.conj() @ Rs @ h_flat).real) + sigma2_c))

    # ---- SDP formulation ----
    # max v^H M_tot v + 2 Re(l_tot^H v)
    # s.t. v^H M_sinr v + 2 Re(l_sinr^H v) + const_sinr >= 0
    #      |v_n| = 1  ->  diag(V) = 1

    # Augmented matrix approach: [v; 1] [v; 1]^H
    # v^H M v + 2 Re(l^H v) + c = tr(M_aug V_aug)

    # For objective:
    M_obj_aug = np.zeros((N_irs + 1, N_irs + 1), dtype=complex)
    M_obj_aug[:N_irs, :N_irs] = M_tot
    M_obj_aug[:N_irs, N_irs] = l_tot
    M_obj_aug[N_irs, :N_irs] = l_tot.conj()
    M_obj_aug[N_irs, N_irs] = const_obj

    # For SINR constraint:
    M_sinr_aug = np.zeros((N_irs + 1, N_irs + 1), dtype=complex)
    M_sinr_aug[:N_irs, :N_irs] = M_sinr
    M_sinr_aug[:N_irs, N_irs] = l_sinr
    M_sinr_aug[N_irs, :N_irs] = l_sinr.conj()
    M_sinr_aug[N_irs, N_irs] = const_sinr

    # ---- Solve SDP ----
    V_var = cp.Variable((N_irs + 1, N_irs + 1), complex=True)

    constraints = [
        V_var >> 0,                             # PSD
        cp.real(cp.trace(V_var)) <= N_irs + 1,  # normalize
        V_var[N_irs, N_irs] == 1,                # last element = 1 (augmented)
    ]

    # SINR constraint: tr(M_sinr_aug V_aug) >= 0
    constraints.append(
        cp.real(cp.trace(M_sinr_aug @ V_var)) >= 0
    )

    # Objective: maximize tr(M_obj_aug V_aug)
    obj = cp.Maximize(cp.real(cp.trace(M_obj_aug @ V_var)))

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=5000)
    except Exception as e:
        return None, {"status": f"SDP solver error: {e}"}

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, {"status": f"SDP infeasible: {prob.status}"}

    V_opt = V_var.value  # (N_irs+1) x (N_irs+1)
    V_irs = V_opt[:N_irs, :N_irs]  # N_irs x N_irs

    # ---- Randomization: recover rank-1 v from V ----
    v_best = _randomization(V_irs, Rc, Rs, A_mat, a_flat,
                            b_flat, b_dot,     # <-- FIX: pass b_flat and b_dot
                            norm_b_sq, norm_bdot_sq,
                            alpha_sq, sigma2_s, T, trials)

    # ---- Compute CRB with best v ----
    a_eff = a_flat + A_mat @ v_best
    crb = _compute_crb_given_aeff(a_eff, Rc, Rs, b_flat, b_dot,
                                   alpha_sq, sigma2_s, T)

    return v_best, {
        "status": prob.status,
        "crb": crb,
        "SDP_obj": prob.value,
        "trials_used": trials
    }


def _randomization(V, Rc, Rs, A_mat, a_flat,
                   b_flat, b_dot_flat,          # <-- FIX: added b_flat and b_dot_flat params
                   norm_b_sq, norm_bdot_sq,
                   alpha_sq, sigma2_s, T, trials=100):
    """
    SDR randomization: sample candidates from V, pick best.

    Args:
        V: SDP solution (N_irs x N_irs)
        b_flat: Flattened RX steering vector (Mr,)
        b_dot_flat: Flattened RX steering derivative (Mr,)
        Other params needed to evaluate CRB for each candidate.

    Returns:
        v_best: Best phase shift vector (N_irs,), |v_best[n]| = 1
    """
    N = V.shape[0]
    best_crb = float('inf')
    v_best = np.ones(N, dtype=complex)  # default: zero phase shift

    # Cholesky-like decomposition: V approx LL^H
    try:
        # Add small regularization for numerical stability
        V_reg = V + 1e-8 * np.eye(N, dtype=complex)
        L = np.linalg.cholesky(V_reg)
    except np.linalg.LinAlgError:
        # Fallback: use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(V)
        eigvals = np.maximum(eigvals, 0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    for _ in range(trials):
        # Sample random Gaussian vector
        xi = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        v_tilde = L @ xi
        # Project to unit circle
        v = v_tilde / (np.abs(v_tilde) + 1e-15)

        # Evaluate CRB
        a_eff = a_flat + A_mat @ v
        crb = _compute_crb_given_aeff(a_eff, Rc, Rs,
                                       b_flat, b_dot_flat,  # <-- FIX: pass proper b vectors
                                       alpha_sq, sigma2_s, T)
        if crb < best_crb:
            best_crb = crb
            v_best = v.copy()

    return v_best


def _compute_crb_given_aeff(a_eff, Rc, Rs,
                            b_flat, b_dot_flat,
                            alpha_sq, sigma2_s, T):
    """
    Compute CRB given effective steering vector a_eff.  [internal]
    Same as compute_crb_case2 formula (Eq.45).

    Args:
        a_eff: Effective steering vector (Mt,)
        Rc: Information covariance (Mt x Mt)
        Rs: Sensing covariance (Mt x Mt)
        b_flat: RX steering vector (Mr,)
        b_dot_flat: RX steering derivative (Mr,)
        alpha_sq: |alpha|^2 target coefficient
        sigma2_s: Sensing RX noise power
        T: Number of symbols

    Returns:
        crb: CRB value (rad^2)
    """
    aH_Rc_a = float((a_eff.conj() @ Rc @ a_eff).real)
    aH_Rs_a = float((a_eff.conj() @ Rs @ a_eff).real)

    if aH_Rc_a <= 1e-20 and aH_Rs_a <= 1e-20:
        return 1e10

    norm_b_sq = float(np.linalg.norm(b_flat) ** 2)
    norm_bdot_sq = float(np.linalg.norm(b_dot_flat) ** 2)

    gamma_ran = alpha_sq * aH_Rc_a * norm_b_sq / sigma2_s

    A_s = aH_Rs_a * norm_bdot_sq
    A_c = aH_Rc_a * norm_bdot_sq

    if gamma_ran > 0:
        F = A_s + (gamma_ran / (1 + gamma_ran)) * A_c
    else:
        F = A_s

    if F <= 1e-20:
        return 1e10

    return sigma2_s / (2 * T * alpha_sq * F)
