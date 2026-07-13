"""
IRS 相移优化求解器 — SDR (Semidefinite Relaxation) — Active IRS 版
=================================================================
固定 R_c, R_s 后，优化 Θ = diag(v) 以最小化 CRB。
Active IRS 支持：|v[n]| ≤ A_MAX（而非 passive 的 |v[n]| = 1）

求解步骤:
  1. 将目标函数和 SINR 约束写成 v 的二次型
  2. 松弛 rank-1 约束 → SDP (V ⪰ 0, diag(V) ≤ A_MAX²)
  3. CVXPY 求解
  4. 随机化恢复 rank-1 解（投影到 |v| ≤ A_MAX）
"""

import numpy as np
import cvxpy as cp
from channels import compute_effective_a, compute_effective_h
from config import AO_MAX_ITER, AO_TOL, SDR_TRIALS, A_MAX, P_RIS, SIGMA2_RIS
from sca_solver import solve_p4_sca
from crb import compute_crb_irs


def _build_a_eff_linear_map(G, h_r, N_irs, Mt):
    h_r_flat = h_r.flatten()
    A = G.T @ np.diag(h_r_flat)
    return A


def _build_h_eff_linear_map(G, h_rc, N_irs, Mt):
    h_rc_flat = h_rc.flatten()
    C = G.T @ np.diag(h_rc_flat)
    return C


def solve_irs_sdr(Rc, Rs, a, h, G, h_r, h_rc,
                  b, b_dot, alpha_sq,
                  sigma2_c, sigma2_s, gamma_0,
                  T, N_irs, Mt, trials=100, active=True):
    """
    Solve IRS phase shift optimization via SDR (Active or Passive).

    Active IRS: |v[n]| ≤ A_MAX (amplitude constraint)
    Passive IRS: |v[n]| = 1 (unit modulus)
    """
    a_flat = a.flatten()
    h_flat = h.flatten()
    b_flat = b.flatten()

    norm_b_sq = np.linalg.norm(b_flat) ** 2
    norm_bdot_sq = np.linalg.norm(b_dot) ** 2

    # ---- Build linear maps ----
    A_mat = _build_a_eff_linear_map(G, h_r, N_irs, Mt)
    C_mat = _build_h_eff_linear_map(G, h_rc, N_irs, Mt)

    # ---- Precompute quadratic form matrices ----
    aH_Rc_a = float((a_flat.conj() @ Rc @ a_flat).real)
    aH_Rs_a = float((a_flat.conj() @ Rs @ a_flat).real)

    M_Rc = A_mat.conj().T @ Rc @ A_mat
    M_Rs = A_mat.conj().T @ Rs @ A_mat
    M_hRc = C_mat.conj().T @ Rc @ C_mat
    M_hRs = C_mat.conj().T @ Rs @ C_mat

    l_Rc = A_mat.conj().T @ Rc @ a_flat
    l_Rs = A_mat.conj().T @ Rs @ a_flat
    l_hRc = C_mat.conj().T @ Rc @ h_flat
    l_hRs = C_mat.conj().T @ Rs @ h_flat

    # ---- Fixed weight w for objective ----
    gamma_ran = alpha_sq * aH_Rc_a * norm_b_sq / sigma2_s
    w = gamma_ran / (1 + gamma_ran) if gamma_ran > 0 else 0

    R_tot = Rs + w * Rc
    M_tot = A_mat.conj().T @ R_tot @ A_mat
    l_tot = A_mat.conj().T @ R_tot @ a_flat
    const_obj = float((a_flat.conj() @ R_tot @ a_flat).real)

    # ---- SINR constraint (cross-multiplied) ----
    M_sinr = M_hRc - gamma_0 * M_hRs
    l_sinr = l_hRc - gamma_0 * l_hRs
    const_sinr = (float((h_flat.conj() @ Rc @ h_flat).real)
                  - gamma_0 * (float((h_flat.conj() @ Rs @ h_flat).real) + sigma2_c))

    # ---- Augmented matrices ----
    M_obj_aug = np.zeros((N_irs + 1, N_irs + 1), dtype=complex)
    M_obj_aug[:N_irs, :N_irs] = M_tot
    M_obj_aug[:N_irs, N_irs] = l_tot
    M_obj_aug[N_irs, :N_irs] = l_tot.conj()
    M_obj_aug[N_irs, N_irs] = const_obj

    M_sinr_aug = np.zeros((N_irs + 1, N_irs + 1), dtype=complex)
    M_sinr_aug[:N_irs, :N_irs] = M_sinr
    M_sinr_aug[:N_irs, N_irs] = l_sinr
    M_sinr_aug[N_irs, :N_irs] = l_sinr.conj()
    M_sinr_aug[N_irs, N_irs] = const_sinr

    # ---- Active RIS power constraint matrix ----
    # RIS input power: tr(Θ^H Θ (G(Rc+Rs)G^H + σ²_RIS I))
    # ≈ v^H (diag(G(Rc+Rs)G^H) + σ²_RIS I) v
    Rx = Rc + Rs
    G_Rx_GH = G @ Rx @ G.conj().T  # N_irs x N_irs
    D = np.diag(np.diag(G_Rx_GH).real) + SIGMA2_RIS * np.eye(N_irs)
    M_power = D  # N_irs x N_irs (diagonal)

    # ---- Solve SDP ----
    V_var = cp.Variable((N_irs + 1, N_irs + 1), complex=True)

    constraints = [
        V_var >> 0,                              # PSD
        V_var[N_irs, N_irs] == 1,                 # augmented element = 1
    ]

    # Amplitude constraint: |v[n]|² ≤ A_MAX² (active) or |v[n]|² = 1 (passive)
    if active:
        # Active RIS: |v[n]| ≤ A_MAX → diag(V) ≤ A_MAX²
        constraints.append(cp.diag(V_var)[:N_irs] <= A_MAX ** 2)
        # Active RIS power constraint: tr(D @ V[:N_irs,:N_irs]) ≤ P_RIS
        constraints.append(cp.real(cp.trace(M_power @ V_var[:N_irs, :N_irs])) <= P_RIS)
    else:
        # Passive IRS: |v[n]| = 1 → diag(V) = 1
        constraints.append(cp.diag(V_var)[:N_irs] == 1)
        constraints.append(cp.real(cp.trace(V_var)) <= N_irs + 1)

    # SINR constraint
    constraints.append(
        cp.real(cp.trace(M_sinr_aug @ V_var)) >= 0
    )

    obj = cp.Maximize(cp.real(cp.trace(M_obj_aug @ V_var)))

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=5000)
    except Exception as e:
        return None, {"status": f"SDP solver error: {e}"}

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, {"status": f"SDP infeasible: {prob.status}"}

    V_opt = V_var.value
    V_irs = V_opt[:N_irs, :N_irs]

    # ---- Randomization ----
    v_best = _randomization(V_irs, Rc, Rs, A_mat, a_flat,
                            b_flat, b_dot,
                            norm_b_sq, norm_bdot_sq,
                            alpha_sq, sigma2_s, T, trials,
                            active=active)

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
                   b_flat, b_dot_flat,
                   norm_b_sq, norm_bdot_sq,
                   alpha_sq, sigma2_s, T, trials=100, active=True):
    """
    SDR randomization with projection.
    Passive: project to |v[n]| = 1
    Active:  project to |v[n]| ≤ A_MAX
    """
    N = V.shape[0]
    best_crb = float('inf')
    v_best = np.ones(N, dtype=complex)

    try:
        V_reg = V + 1e-8 * np.eye(N, dtype=complex)
        L = np.linalg.cholesky(V_reg)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(V)
        eigvals = np.maximum(eigvals, 0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    for _ in range(trials):
        xi = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        v_tilde = L @ xi

        if active:
            # Active RIS: clip to |v[n]| ≤ A_MAX
            mag = np.abs(v_tilde)
            v = np.where(mag > A_MAX, A_MAX * v_tilde / (mag + 1e-15), v_tilde)
        else:
            # Passive IRS: project to unit circle
            v = v_tilde / (np.abs(v_tilde) + 1e-15)

        a_eff = a_flat + A_mat @ v
        crb = _compute_crb_given_aeff(a_eff, Rc, Rs,
                                       b_flat, b_dot_flat,
                                       alpha_sq, sigma2_s, T)
        if crb < best_crb:
            best_crb = crb
            v_best = v.copy()

    return v_best


def _compute_crb_given_aeff(a_eff, Rc, Rs,
                            b_flat, b_dot_flat,
                            alpha_sq, sigma2_s, T):
    """Compute CRB given effective steering vector. (same as before)"""
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


# ========================================================================
# AO 交替优化框架（兼容 Active/Passive IRS）
# ========================================================================
def ao_optimize(gamma_0, a_eff_init, h_eff_init, G, h_r, h_rc,
                b, b_dot, alpha_sq,
                sigma2_c, sigma2_s, P, Mt, Mr, T, N_irs,
                direct_blocked=False, a_dir=None, active=True):
    """
    Alternating optimization for IRS-assisted ISAC.
    active=True  → Active IRS (amplify)
    active=False → Passive IRS (unit modulus)
    """
    v = np.ones(N_irs, dtype=complex)
    if active:
        v = v * (A_MAX / np.sqrt(2))  # initial amplitude for active IRS
    a_eff = a_eff_init.copy()
    h_eff = h_eff_init.copy()

    history = []

    for k in range(AO_MAX_ITER):
        Rc, Rs, info_sca = solve_p4_sca(gamma_0, h_eff, a_eff, sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq)
        if Rc is None:
            return None, None, None, {"status": f"SCA failed at AO iter {k}"}

        a_sdr = a_dir if a_dir is not None and not direct_blocked else np.zeros(Mt, dtype=complex)
        v_new, info_irs = solve_irs_sdr(
            Rc, Rs, a_sdr, h_eff, G, h_r, h_rc, b, b_dot, alpha_sq,
            sigma2_c, sigma2_s, gamma_0, T, N_irs, Mt, trials=SDR_TRIALS,
            active=active
        )
        if v_new is None:
            v_new = v

        a_eff_new = compute_effective_a(a_dir if a_dir is not None else a_eff, G, h_r, v_new, direct_blocked=direct_blocked)
        h_eff_new = compute_effective_h(h_eff, G, h_rc, v_new)

        crb_old = compute_crb_irs(Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)
        crb_new = compute_crb_irs(Rc, Rs, a_eff_new, b, b_dot, alpha_sq, sigma2_s, T)

        history.append({"iter": k, "crb": crb_new})
        crb_change = abs(crb_new - crb_old) / (abs(crb_old) + 1e-15)
        v, a_eff, h_eff = v_new, a_eff_new, h_eff_new

        if crb_change < AO_TOL and k > 0:
            break

    return Rc, Rs, v, {"status": f"converged in {k+1} AO iters", "history": history}
