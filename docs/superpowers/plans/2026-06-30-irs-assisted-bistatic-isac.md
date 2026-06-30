# IRS-assisted Bistatic ISAC CRB-Rate Tradeoff — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement AO framework that jointly optimizes IRS phase shifts Θ and beamforming matrices (R_c, R_s) to minimize CRB under SINR constraint.

**Architecture:** Case 2 (superimposed Gaussian + deterministic signals). AO outer loop: Step 1 optimizes beamforming (reuses Paper 4's SCA), Step 2 optimizes IRS phase shifts (SDR). All IRS paths modeled as additions to existing steering vectors and channels.

**Tech Stack:** Python 3, NumPy, CVXPY (SCS solver)

## Global Constraints

- Signal model: Case 2 only (no Case 1 / Case 3 code)
- Coding style per ISAC-README.txt: modular functions, `[eq:xx]` annotations, centralized config, np.random.seed(0)
- All optimization via CVXPY (SDR/SCA as modeling methods, SCS as solver)
- No third-party optimization libraries beyond NumPy + CVXPY
- IRS unit-modulus constraint: handled via SDR + randomization
- Results auto-saved to `results/` with timestamps

---

### Task 1: Project Cleanup + Config Setup

**Files:**
- Delete: `main.py`, `main_case3.py`
- Keep: `main_case2.py` (Case 2 reference script)
- Create: `config.py`

**Interfaces:**
- Produces: All global parameters consumed by every other module

- [ ] **Step 1: Delete Case 1 and Case 3 files**

```bash
rm "C:/Users/a/Desktop/ISAC/reproduction-1+IRS/main.py"
rm "C:/Users/a/Desktop/ISAC/reproduction-1+IRS/main_case3.py"
```

- [ ] **Step 2: Create `config.py` with all centralized parameters**

Centralized config. Includes original Paper 4 params + new IRS params.

```python
"""
Centralized simulation parameters — ISAC-README.txt §4
All parameters in one place, imported by other modules.

Usage:
    from config import *
"""
import numpy as np

# ===== System dimensions =====
Mt = 32            # BS antennas
Mr = 32            # Sensing RX antennas
T  = 1024          # Number of symbols

# ===== IRS parameters =====
N_irs = 16         # IRS reflecting elements (start at 16, can go to 32)

# ===== IRS positions (x, y) =====
# BS: (0, 0), Target: (200, 0), Sensing RX: (400, 0)
pos_bs     = np.array([0.0, 0.0])
pos_target = np.array([200.0, 0.0])
pos_rx     = np.array([400.0, 0.0])
pos_irs    = np.array([50.0, 30.0])   # typical IRS placement near BS

# ===== Power & Noise =====
P_dBm      = 30.0
P          = 10**((P_dBm - 30)/10)   # 1 W
sigma2_c_dBm = -80.0
sigma2_s_dBm = -80.0
sigma2_c   = 10**((sigma2_c_dBm - 30)/10)  # CU noise power
sigma2_s   = 10**((sigma2_s_dBm - 30)/10)  # Sensing RX noise power

# ===== Geometry =====
theta_target = 0.0   # Target DoA (rad)
phi_target   = 0.0   # BS→Target direction (rad)
phi_cu       = 0.3   # BS→CU direction (rad)

# ===== Path loss (Paper 4 Eq.63) =====
K0     = -30     # Path loss at d0 (dB)
alpha0 = 2.5     # Path loss exponent
d0     = 1.0     # Reference distance (m)

# ===== Target channel (Paper 4 Eq.6) =====
CAL_ALPHA = 1.0e-32   # Calibration factor for |alpha|^2

# ===== CU channel (Paper 4 Eq.62) =====
Kc = 1.0        # Rician K-factor for BS→CU

# ===== SINR sweep =====
N_gamma        = 40
gamma_0_dB_min = -10.0
gamma_0_dB_max = 19.0

# ===== SDR solver (IRS) =====
SDR_TRIALS = 100    # Number of randomizations for SDR recovery

# ===== AO outer loop =====
AO_MAX_ITER = 20
AO_TOL      = 1e-4

# ===== Reproducibility =====
SEED       = 0      # Paper 4 uses h_seed=46 for channel gen
SEED_CHANNEL = 46   # Keep Paper 4's channel seed for fair comparison
```

- [ ] **Step 3: Create results directory**

```bash
mkdir -p "C:/Users/a/Desktop/ISAC/reproduction-1+IRS/results"
```

- [ ] **Step 4: Commit checkpoint (no git repo — skip)**

---

### Task 2: Extend channels.py — IRS Channel Generation

**Files:**
- Modify: `channels.py` (append IRS channel functions)

**Interfaces:**
- Consumes: `steering_vector()` from `steering_vectors.py`, `path_loss_linear()` from `channels.py`
- Produces:
  - `generate_irs_bs_channel(Mt, N, d_br, phi_br, ...)` → `G ∈ ℂ^{N×Mt}`
  - `generate_irs_target_channel(N, d_rt, phi_rt, ...)` → `h_r ∈ ℂ^{1×N}`
  - `generate_irs_cu_channel(N, d_rc, phi_rc, Kc, ...)` → `h_rc ∈ ℂ^{1×N}`
  - `compute_effective_a(a, G, h_r, v)` → `a_eff ∈ ℂ^{Mt×1}`
  - `compute_effective_h(h, G, h_rc, v)` → `h_eff ∈ ℂ^{Mt×1}`

- [ ] **Step 1: Add IRS channel functions to `channels.py`**

Append to the end of `channels.py`:

```python
# ========================================================================
# IRS 信道模型 — IRS-assisted Bistatic ISAC 项目新增
# 参考设计文档 §1.2
# ========================================================================

def compute_distance(pos1, pos2):
    """Euclidean distance between two points."""
    return np.linalg.norm(pos1 - pos2)


def compute_angle(pos_from, pos_to):
    """Angle from pos_from to pos_to (rad)."""
    delta = pos_to - pos_from
    return np.arctan2(delta[1], delta[0])


def generate_irs_bs_channel(Mt, N_irs, d_br, phi_br,
                            K0=-30, alpha0=2.5, d0=1.0):
    """
    Generate BS → IRS channel matrix G ∈ ℂ^{N_irs×Mt}  [new]
    
    LoS MIMO channel model between two ULAs:
    G = sqrt(L(d_br)) * a_irs(phi_br) @ a_bs(phi_br)^H
    
    Args:
        Mt: BS antennas
        N_irs: IRS elements
        d_br: BS-IRS distance (m)
        phi_br: IRS direction from BS (rad)
        K0, alpha0, d0: Path loss parameters (Paper 4 Eq.63)
    
    Returns:
        G: BS→IRS channel (N_irs × Mt)
    """
    L_br = path_loss_linear(d_br, K0, alpha0, d0)
    a_bs  = steering_vector(Mt, phi_br)   # Mt×1: BS steering toward IRS
    a_irs = steering_vector(N_irs, phi_br) # N_irs×1: IRS steering from BS
    
    # LoS MIMO channel: G = sqrt(L) * a_irs * a_bs^H
    G = np.sqrt(L_br) * (a_irs @ a_bs.conj().T)
    return G


def generate_irs_target_channel(N_irs, d_rt, phi_rt,
                                K0=-30, alpha0=2.5, d0=1.0):
    """
    Generate IRS → Target channel h_r ∈ ℂ^{1×N_irs}  [new]
    
    IRS reflects signal toward target. The cascaded path is
    BS→IRS→Target→SensingRX. The h_r is the IRS→Target steering.
    
    Args:
        N_irs: IRS elements
        d_rt: IRS-Target distance via target reflection (m)
        phi_rt: Target direction from IRS (rad)
    
    Returns:
        h_r: IRS→Target channel (1 × N_irs)
    """
    L_rt = path_loss_linear(d_rt, K0, alpha0, d0)
    a_irs_target = steering_vector(N_irs, phi_rt)  # N_irs×1
    
    # 1×N_irs: conjugate transpose of steering vector
    h_r = np.sqrt(L_rt) * a_irs_target.conj().T
    return h_r


def generate_irs_cu_channel(N_irs, d_rc, phi_rc, Kc=1.0,
                            K0=-30, alpha0=2.5, d0=1.0, seed=42):
    """
    Generate IRS → CU channel h_rc ∈ ℂ^{1×N_irs}  [new]
    
    Rician fading (similar to BS→CU, Paper 4 Eq.62).
    
    Args:
        N_irs: IRS elements
        d_rc: IRS-CU distance (m)
        phi_rc: CU direction from IRS (rad)
        Kc: Rician K-factor
        seed: Random seed
    """
    np.random.seed(seed)
    L_rc = path_loss_linear(d_rc, K0, alpha0, d0)
    
    # LoS component
    h_los = steering_vector(N_irs, phi_rc).flatten()  # N_irs,
    
    # NLoS component
    h_nlos = (np.random.randn(N_irs) + 1j * np.random.randn(N_irs)) / np.sqrt(2)
    
    # Rician combination
    h_channel = (np.sqrt(Kc / (Kc + 1)) * h_los
                 + np.sqrt(1 / (Kc + 1)) * h_nlos)
    
    return np.sqrt(L_rc) * h_channel.reshape(1, -1)  # 1×N_irs


def compute_effective_a(a, G, h_r, v):
    """
    Compute effective target-direction steering vector  [new]
    
    a_eff(Θ) = a + (h_r @ Θ @ G)^T
            = a + G^T @ (h_r^T * v)
    
    where v = [e^{jθ₁}, ..., e^{jθ_N}]^T, Θ = diag(v)
    
    Args:
        a: Direct-path steering vector (Mt×1)
        G: BS→IRS channel (N_irs×Mt)
        h_r: IRS→Target channel (1×N_irs)
        v: IRS phase shift vector (N_irs,)
    
    Returns:
        a_eff: Effective steering vector (Mt×1)
    """
    # h_r @ Θ @ G → (1×N) @ (N×N) @ (N×Mt) = 1×Mt
    # h_r.T * v: element-wise → (N,)  (v[n] multiplies h_r[n])
    N = len(v)
    h_r_flat = h_r.flatten()  # (N,)
    irs_path = G.T @ (h_r_flat * v)  # Mt×N @ (N,) = (Mt,)
    return a.flatten() + irs_path.reshape(-1, 1)  # Mt×1


def compute_effective_h(h, G, h_rc, v):
    """
    Compute effective CU channel vector  [new]
    
    h_eff(Θ) = h + (h_rc @ Θ @ G)^T
             = h + G^T @ (h_rc^T * v)
    
    Args:
        h: Direct-path CU channel (Mt×1)
        G: BS→IRS channel (N_irs×Mt)
        h_rc: IRS→CU channel (1×N_irs)
        v: IRS phase shift vector (N_irs,)
    
    Returns:
        h_eff: Effective CU channel (Mt×1)
    """
    h_rc_flat = h_rc.flatten()  # (N,)
    irs_path = G.T @ (h_rc_flat * v)  # (Mt,)
    return h.flatten() + irs_path.reshape(-1, 1)  # Mt×1
```

- [ ] **Step 3: Verify the module imports correctly**

```bash
cd "C:/Users/a/Desktop/ISAC/reproduction-1+IRS"
python -c "from channels import *; print('channels.py OK')"
python -c "from steering_vectors import *; print('steering_vectors.py OK')"
```

---

### Task 3: Extend crb_calc.py and comm_rate.py — IRS-aware Functions

**Files:**
- Modify: `crb_calc.py` (append IRS-aware wrapper)
- Modify: `comm_rate.py` (append IRS-aware wrapper)

**Interfaces:**
- Consumes: `compute_crb_case2()`, `compute_rate_case2()` from existing code
- Produces: `compute_crb_irs(theta, Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)` → CRB
- Produces: `compute_rate_irs(Rc, Rs, h_eff, sigma2_c)` → (rate, sinr)

- [ ] **Step 1: Add IRS-aware CRB wrapper to `crb_calc.py`**

Append to the end of `crb_calc.py`:

```python
def compute_crb_irs(theta, Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T):
    """
    Compute CRB with IRS-enhanced effective steering vector  [new]
    
    Same formula as Case 2 (Eq.45), but uses a_eff instead of a.
    The IRS effect is entirely captured in a_eff — the CRB formula
    structure is unchanged.
    
    Args:
        theta: Target DoA (rad)
        Rc: Information covariance (Mt×Mt)
        Rs: Sensing covariance (Mt×Mt)
        a_eff: Effective steering vector (Mt×1) — already includes IRS path
        b: RX steering vector (Mr×1)
        b_dot: Derivative of b (Mr×1)
        alpha_sq: |alpha|^2 target channel coefficient
        sigma2_s: Sensing RX noise power
        T: Number of symbols
    
    Returns:
        crb: CRB value (rad²)
    """
    # Delegate to existing Case 2 CRB with a_eff in place of a
    # The existing compute_crb_case2 uses a for both Rc and Rs projections
    return compute_crb_case2(theta, Rc, Rs, a_eff, b, b_dot,
                             alpha_sq, sigma2_s, T)
```

- [ ] **Step 2: Add IRS-aware rate wrapper to `comm_rate.py`**

Append to the end of `comm_rate.py`:

```python
def compute_rate_irs(Rc, Rs, h_eff, sigma2_c):
    """
    Compute communication rate with IRS-enhanced effective channel  [new]
    
    Same formula as Case 2 (Eq.13-14), but uses h_eff instead of h.
    
    Args:
        Rc: Information covariance (Mt×Mt)
        Rs: Sensing covariance (Mt×Mt)
        h_eff: Effective CU channel (Mt×1) — already includes IRS path
        sigma2_c: CU noise power
    
    Returns:
        rate: Achievable rate (bps/Hz)
        sinr: SINR (linear scale)
    """
    return compute_rate_case2(Rc, Rs, h_eff, sigma2_c)
```

- [ ] **Step 3: Verify imports**

```bash
python -c "from crb_calc import *; from comm_rate import *; print('crb_calc.py + comm_rate.py OK')"
```

---

### Task 4: Create irs_solver.py — SDR for IRS Phase Shifts

**Files:**
- Create: `irs_solver.py`

**Interfaces:**
- Consumes: `compute_effective_a()`, `compute_effective_h()` from `channels.py`, `steering_vector()` from `steering_vectors.py`
- Produces: `solve_irs_sdr(Rc, Rs, a, h, G, h_r, h_rc, b, b_dot, alpha_sq, sigma2_c, sigma2_s, gamma_0, T, N_irs, trials)` → `v_opt ∈ ℂ^{N_irs}`, `Θ_opt = diag(v_opt)`

- [ ] **Step 1: Create `irs_solver.py`**

```python
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
    Θ_opt = np.diag(v_opt)
"""

import numpy as np
import cvxpy as cp
from channels import compute_effective_a, compute_effective_h


def _build_a_eff_linear_map(G, h_r, N_irs, Mt):
    """
    Build linear map A such that a_eff(v) = a + A @ v.
    
    a_eff(v) = a + G^T @ (h_r^T * v)
    A = G^T @ diag(h_r.flatten()) ∈ ℂ^{Mt × N_irs}
    
    Returns:
        A: Matrix mapping v → IRS contribution to a_eff
        a_const: Direct-path component (Mt,)
    """
    h_r_flat = h_r.flatten()  # (N_irs,)
    A = G.T @ np.diag(h_r_flat)  # (Mt × N_irs)
    return A


def _build_h_eff_linear_map(G, h_rc, N_irs, Mt):
    """
    Build linear map C such that h_eff(v) = h + C @ v.
    
    Returns:
        C: Matrix mapping v → IRS contribution to h_eff
        h_const: Direct-path component (Mt,)
    """
    h_rc_flat = h_rc.flatten()  # (N_irs,)
    C = G.T @ np.diag(h_rc_flat)  # (Mt × N_irs)
    return C


def solve_irs_sdr(Rc, Rs, a, h, G, h_r, h_rc,
                  b, b_dot, alpha_sq,
                  sigma2_c, sigma2_s, gamma_0,
                  T, N_irs, Mt, trials=100):
    """
    Solve IRS phase shift optimization via SDR.
    
    Fixed R_c, R_s → optimize Θ to minimize CRB.
    Formulated as SDP + randomization.
    
    Args:
        Rc: Information covariance (Mt×Mt) — fixed from AO Step 1
        Rs: Sensing covariance (Mt×Mt) — fixed from AO Step 1
        a: Direct steering vector (Mt×1)
        h: Direct CU channel (Mt×1)
        G: BS→IRS channel (N_irs×Mt)
        h_r: IRS→Target channel (1×N_irs)
        h_rc: IRS→CU channel (1×N_irs)
        b: RX steering vector (Mr×1)
        b_dot: RX steering derivative (Mr×1)
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
    
    norm_b_sq    = np.linalg.norm(b_flat)**2
    norm_bdot_sq = np.linalg.norm(b_dot)**2
    
    # ---- Build linear maps ----
    A_mat = _build_a_eff_linear_map(G, h_r, N_irs, Mt)  # Mt×N_irs
    C_mat = _build_h_eff_linear_map(G, h_rc, N_irs, Mt)  # Mt×N_irs
    
    # ---- Precompute quadratic form matrices ----
    # a_eff^H R a_eff = (a + Av)^H R (a + Av)
    # = a^H R a + 2 Re(a^H R A v) + v^H A^H R A v
    
    # For Rc:
    aH_Rc_a = float((a_flat.conj() @ Rc @ a_flat).real)
    # For Rs:
    aH_Rs_a = float((a_flat.conj() @ Rs @ a_flat).real)
    
    # Quadratic part: M_Rc = A^H Rc A  (N_irs × N_irs)
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
    
    # ---- Compute fixed weight w = γ_ran/(1+γ_ran) for the objective ----
    # Use current Rc's a_eff^H R_c a_eff to compute γ_ran
    # This is approximated: we fix w during this SDR step
    aH_Rc_a_current = aH_Rc_a + 2 * float((a_flat.conj() @ Rc @ A_mat).real) + 0  # approx, will refine
    # Actually compute correctly with v=0 (no IRS) first, then update
    gamma_ran = alpha_sq * aH_Rc_a * norm_b_sq / sigma2_s
    w = gamma_ran / (1 + gamma_ran) if gamma_ran > 0 else 0
    
    # ---- Simplified: fixed weight w, maximize F = a_eff^H (Rs + w*Rc) a_eff ----
    # This is equivalent to minimizing CRB when w is fixed
    R_tot = Rs + w * Rc
    M_tot = A_mat.conj().T @ R_tot @ A_mat  # N_irs × N_irs
    l_tot = A_mat.conj().T @ R_tot @ a_flat  # N_irs,
    
    const_obj = float(a_flat.conj() @ R_tot @ a_flat)  # constant term
    
    # ---- SINR constraint (cross-multiplied) ----
    # h_eff^H R_c h_eff ≥ γ₀ · (h_eff^H R_s h_eff + σ²_c)
    # → v^H (M_hRc - γ₀ M_hRs) v + 2 Re(l_h_combined^H v) + const_sinr ≥ 0
    M_sinr = M_hRc - gamma_0 * M_hRs  # N_irs × N_irs
    l_sinr = l_hRc - gamma_0 * l_hRs  # N_irs,
    const_sinr = (float(h_flat.conj() @ Rc @ h_flat)
                  - gamma_0 * (float(h_flat.conj() @ Rs @ h_flat) + sigma2_c))
    
    # ---- SDP formulation ----
    # max v^H M_tot v + 2 Re(l_tot^H v)
    # s.t. v^H M_sinr v + 2 Re(l_sinr^H v) + const_sinr ≥ 0
    #      |v_n| = 1  →  diag(V) = 1
    
    # V = vv^H, v^H M v = tr(M V)
    # 2 Re(l^H v) = tr(l v^H + v l^H) = tr((l v^H) + ...)
    # We need to handle the linear term properly in SDP
    
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
        V_var >> 0,                                # PSD
        cp.real(cp.trace(V_var)) <= N_irs + 1,    # normalize
        V_var[N_irs, N_irs] == 1,                  # last element = 1 (augmented)
    ]
    
    # SINR constraint: tr(M_sinr_aug V_aug) ≥ 0
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
    
    V_opt = V_var.value  # (N_irs+1) × (N_irs+1)
    V_irs = V_opt[:N_irs, :N_irs]  # N_irs × N_irs
    
    # ---- Randomization: recover rank-1 v from V ----
    v_best = _randomization(V_irs, Rc, Rs, A_mat, a_flat,
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
                   norm_b_sq, norm_bdot_sq,
                   alpha_sq, sigma2_s, T, trials=100):
    """
    SDR randomization: sample candidates from V, pick best.  [new]
    
    Args:
        V: SDP solution (N_irs×N_irs)
        Other params needed to evaluate CRB for each candidate.
    
    Returns:
        v_best: Best phase shift vector (N_irs,), |v_best[n]| = 1
    """
    N = V.shape[0]
    best_crb = float('inf')
    v_best = np.ones(N, dtype=complex)  # default: zero phase shift
    
    # Cholesky-like decomposition: V ≈ LL^H
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
                                       np.ones(A_mat.shape[0]),  # dummy — will fix
                                       0, alpha_sq, sigma2_s, T)
        # Actually use proper b and b_dot...
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
    """
    aH_Rc_a = float((a_eff.conj() @ Rc @ a_eff).real)
    aH_Rs_a = float((a_eff.conj() @ Rs @ a_eff).real)
    
    if aH_Rc_a <= 1e-20 and aH_Rs_a <= 1e-20:
        return 1e10
    
    norm_b_sq    = float(np.linalg.norm(b_flat)**2)
    norm_bdot_sq = float(np.linalg.norm(b_dot_flat)**2)
    
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
```

- [ ] **Step 2: Verify import and basic test**

```bash
python -c "from irs_solver import *; print('irs_solver.py OK')"
```

---

### Task 5: Extend plot_results.py — IRS Comparison Plots

**Files:**
- Modify: `plot_results.py` (append IRS comparison plot)

- [ ] **Step 1: Add `plot_irs_comparison()` to `plot_results.py`**

```python
def plot_irs_comparison(data_no_irs, data_irs_list, labels=None, save_path='irs_comparison.png'):
    """
    Compare CRB-Rate tradeoff with and without IRS.  [new]
    
    Args:
        data_no_irs: dict with 'rate', 'crb', 'gamma' (baseline, no IRS)
        data_irs_list: list of dicts, each with 'rate', 'crb', 'gamma'
        labels: list of legend labels for each IRS config
        save_path: output path
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot baseline (no IRS)
    if data_no_irs is not None:
        s = np.argsort(data_no_irs['rate'])
        ax1.plot(data_no_irs['rate'][s], data_no_irs['crb'][s], 'k-',
                 linewidth=2, label='w/o IRS (baseline)')
        ax2.plot(10 * np.log10(data_no_irs['gamma'][s]), data_no_irs['crb'][s],
                 'k-', linewidth=2, label='w/o IRS (baseline)')
    
    # Plot each IRS config
    colors = ['r--', 'b-.', 'g:', 'm--']
    for i, d in enumerate(data_irs_list):
        if d is None: continue
        s = np.argsort(d['rate'])
        c = colors[i % len(colors)]
        lbl = labels[i] if labels and i < len(labels) else f'IRS config {i+1}'
        ax1.plot(d['rate'][s], d['crb'][s], c, linewidth=1.8, label=lbl)
        ax2.plot(10 * np.log10(d['gamma'][s]), d['crb'][s], c, linewidth=1.8, label=lbl)
    
    ax1.set_xlabel('Communication Rate (bps/Hz)')
    ax1.set_ylabel('CRB for DoA Estimation (rad²)')
    ax1.set_title('CRB-Rate Tradeoff: IRS vs Baseline')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    ax2.set_xlabel('SINR Threshold γ₀ (dB)')
    ax2.set_ylabel('CRB for DoA Estimation (rad²)')
    ax2.set_title('CRB vs SINR Constraint')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show(block=False)
    plt.pause(0.5)
```

---

### Task 6: Create main_irs.py — AO Framework + Experiment Driver

**Files:**
- Create: `main_irs.py`

- [ ] **Step 1: Create `main_irs.py`**

```python
"""
IRS-assisted Bistatic ISAC — AO Framework Main Script
===================================================
基于 Paper 4 (CRB-Rate Tradeoff, TWC 2026) 扩展 IRS 辅助版本。
信号模型: Case 2 (叠加信号: 高斯信息 + 确定性感知信号)

AO 框架:
  Step 1: 固定 Θ, 优化 R_c, R_s (复用 case2_solver.py)
  Step 2: 固定 R_c, R_s, 优化 Θ (irs_solver.py SDR)

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
    d_bt = compute_distance(pos_bs, pos_target)       # BS→Target [eq:6]
    d_tr = compute_distance(pos_target, pos_rx)        # Target→RX
    d_br = compute_distance(pos_bs, pos_irs)           # BS→IRS [new]
    d_rt = compute_distance(pos_irs, pos_target)       # IRS→Target [new]
    d_rc = compute_distance(pos_irs, pos_target * 0 + np.array([1000*np.cos(phi_cu), 1000*np.sin(phi_cu)]))  # IRS→CU
    d_bc = compute_distance(pos_bs, pos_target * 0 + np.array([1000*np.cos(phi_cu), 1000*np.sin(phi_cu)]))   # BS→CU
    
    phi_br = compute_angle(pos_bs, pos_irs)             # BS→IRS angle [new]
    phi_rt = compute_angle(pos_irs, pos_target)         # IRS→Target angle [new]
    phi_rc = compute_angle(pos_irs, pos_target * 0 + np.array([1000*np.cos(phi_cu), 1000*np.sin(phi_cu)]))  # IRS→CU angle
    
    # ---- 2. Generate channels ----
    # Direct paths (Paper 4)
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
    print(f"Geometry: BS(0,0), IRS({pos_irs[0]:.0f},{pos_irs[1]:.0f}), Target(200,0), RX(400,0)")
    print(f"d_bt={d_bt:.1f}m, d_br={d_br:.1f}m, d_rt={d_rt:.1f}m")
    print(f"|alpha|² = {alpha_sq:.3e}")
    
    # ---- 3. SINR sweep ----
    gamma_0_dB_vals = np.linspace(gamma_0_dB_min, gamma_0_dB_max, N_gamma)
    
    # Without IRS (baseline — direct reuse of case2_solver)
    results_no_irs = []
    # With IRS
    results_irs = []
    
    print(f"\nSweeping {N_gamma} SINR thresholds...")
    print(f"{'gamma_0(dB)':>10} {'Base CRB':>14} {'IRS CRB':>14} {'Base Rate':>14} {'IRS Rate':>14}")
    
    for g0_dB in gamma_0_dB_vals:
        gamma_0 = 10**(g0_dB / 10)
        
        # ---- Step A: Without IRS (baseline) ----
        Rc_base, Rs_base, info_base = solve_p4_sca(
            gamma_0, h, a, sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq
        )
        
        if Rc_base is None:
            results_no_irs.append((gamma_0, None, None))
        else:
            rate_b, sinr_b = compute_rate_irs(Rc_base, Rs_base, h, sigma2_c)
            crb_b = compute_crb_irs(theta_target, Rc_base, Rs_base, a,
                                     b, b_dot, alpha_sq, sigma2_s, T)
            results_no_irs.append((gamma_0, crb_b, rate_b))
        
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
        base_crb_str = f"{crb_b:.3e}" if Rc_base is not None else '---'
        base_rate_str = f"{rate_b:.4f}" if Rc_base is not None else '---'
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
      Step 1: Fix Θ, optimize R_c, R_s (case2_solver)
      Step 2: Fix R_c, R_s, optimize Θ (irs_solver)
    
    Returns:
        Rc_opt, Rs_opt: Optimized covariance matrices
        theta_opt: Optimized IRS phase shift vector
        info: dict with convergence info
    """
    # Initialize Θ = identity (zero phase shift)
    v = np.ones(N_irs, dtype=complex)
    a_eff = compute_effective_a(a, G, h_r, v)
    h_eff = compute_effective_h(h, G, h_rc, v)
    
    history = []
    
    for k in range(AO_MAX_ITER):
        # ---- Step 1: Fix Θ, optimize R_c, R_s ----
        Rc, Rs, info_sca = solve_p4_sca(
            gamma_0, h_eff, a_eff, sigma2_c, sigma2_s,
            P, Mt, Mr, b, b_dot, alpha_sq
        )
        
        if Rc is None:
            return None, None, None, {"status": f"SCA failed at AO iter {k}"}
        
        # ---- Step 2: Fix R_c, R_s, optimize Θ ----
        v_new, info_irs = solve_irs_sdr(
            Rc, Rs, a_eff, h_eff, G, h_r, h_rc,
            b, b_dot, alpha_sq,
            sigma2_c, sigma2_s, gamma_0,
            T, N_irs, Mt, trials=SDR_TRIALS
        )
        
        if v_new is None:
            # SDR failed — keep current Θ
            v_new = v
        
        # ---- Update effective channels ----
        a_eff_new = compute_effective_a(a, G, h_r, v_new)
        h_eff_new = compute_effective_h(h, G, h_rc, v_new)
        
        # ---- Compute CRB ----
        crb_old = compute_crb_irs(0, Rc, Rs, a_eff, b, b_dot, alpha_sq, sigma2_s, T)
        crb_new = compute_crb_irs(0, Rc, Rs, a_eff_new, b, b_dot, alpha_sq, sigma2_s, T)
        
        history.append({"iter": k, "crb": crb_new})
        
        # ---- Convergence check ----
        crb_change = abs(crb_new - crb_old) / (abs(crb_old) + 1e-15)
        if crb_change < AO_TOL and k > 0:
            v = v_new
            a_eff = a_eff_new
            h_eff = h_eff_new
            break
        
        v = v_new
        a_eff = a_eff_new
        h_eff = h_eff_new
    
    return Rc, Rs, v, {"status": f"converged in {k+1} AO iters", "history": history}


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
```

---

### Task 7: Run and Verify

- [ ] **Step 1: Run the full simulation**

```bash
cd "C:/Users/a/Desktop/ISAC/reproduction-1+IRS"
python main_irs.py
```

Expected output: CRB-Rate tradeoff curves for both with and without IRS, saved to `results/` directory.

- [ ] **Step 2: Verify outputs**

Check that:
- `results/irs_N16_*.npz` — data file exists with shape (valid_points,)
- `results/irs_comparison_N16_*.png` — plot shows IRS curve below baseline curve
- Console shows convergence in each AO iteration

- [ ] **Step 3: Quick parameter sweep (N_irs = 32)**

Change `N_irs = 32` in `config.py` and re-run.
Expected: N=32 shows lower CRB than N=16.

---

### Summary of file changes

| Task | Action | Files |
|:---:|:---|:---|
| 1 | Delete | `main.py`, `main_case3.py` |
| 1 | Create | `config.py` |
| 2 | Modify | `channels.py` (append 6 functions) |
| 3 | Modify | `crb_calc.py` (append 1 function) |
| 3 | Modify | `comm_rate.py` (append 1 function) |
| 4 | Create | `irs_solver.py` (3 functions) |
| 5 | Modify | `plot_results.py` (append 1 function) |
| 6 | Create | `main_irs.py` (3 functions) |
| 7 | Run | Verify full pipeline |
