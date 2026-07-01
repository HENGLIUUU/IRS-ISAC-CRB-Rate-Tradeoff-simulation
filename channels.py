"""
信道生成函数模块
===============
对应论文 Eq.(6) 和 Eq.(62)-(63)。

用法:
    from channels import generate_rician_channel, compute_alpha_sq
    from steering_vectors import steering_vector

    h = generate_rician_channel(Mt=4, phi_cu=0.3, Kc=1.0, d_bc=1000.0,
                                K0=-30, alpha0=2.5, d0=1.0, seed=42)
    alpha_sq = compute_alpha_sq(d_bt=200.0, d_tr=200.0, beta=1.0,
                                K0=-30, alpha0=2.5, d0=1.0, CAL=3e-30)
"""

import numpy as np
from steering_vectors import steering_vector


def path_loss_linear(d, K0=-30, alpha0=2.5, d0=1.0):
    """
    Path loss model: L(d) = K0 * (d/d0)^(-alpha0)  [Eq.(63)]

    Args:
        d: Distance (m)
        K0: Path loss at d0 (dB), default -30
        alpha0: Path loss exponent, default 2.5
        d0: Reference distance (m), default 1.0

    Returns:
        L: Linear path loss (not dB)
    """
    K0_lin = 10 ** (K0 / 10)
    return K0_lin * (d / d0) ** (-alpha0)


def generate_rician_channel(Mt, phi_cu, Kc=1.0, d_bc=1000.0,
                            K0=-30, alpha0=2.5, d0=1.0, seed=42):
    """
    Generate Rician fading channel for BS → CU link  [Eq.(62)].

    h = sqrt(L) * ( sqrt(K/(K+1)) * h_los + sqrt(1/(K+1)) * h_nlos )

    Args:
        Mt: Number of BS transmit antennas
        phi_cu: CU direction w.r.t. BS (rad)
        Kc: Rician factor (default 1.0)
        d_bc: BS-CU distance in meters (default 1000.0)
        K0, alpha0, d0: Path loss parameters
        seed: Random seed for reproducibility

    Returns:
        h: Channel vector (Mt, 1), complex-valued
    """
    np.random.seed(seed)

    # LoS component: steering vector toward CU direction
    h_los = steering_vector(Mt, phi_cu).flatten()

    # NLoS component: Rayleigh fading
    h_nlos = (np.random.randn(Mt) + 1j * np.random.randn(Mt)) / np.sqrt(2)

    # Rician combination
    h_channel = (np.sqrt(Kc / (Kc + 1)) * h_los
                 + np.sqrt(1 / (Kc + 1)) * h_nlos)

    # Apply path loss
    L_bc = path_loss_linear(d_bc, K0, alpha0, d0)
    h_channel = np.sqrt(L_bc) * h_channel

    return h_channel.reshape(-1, 1)


def compute_alpha_sq(d_bt, d_tr, beta=1.0,
                     K0=-30, alpha0=2.5, d0=1.0, CAL=1.0):
    """
    Compute |alpha|^2 — target channel coefficient  [Eq.(6)].

    alpha^2 = beta / (L(d_BT) * L(d_TR)) * CAL

    where CAL is a calibration factor accounting for RCS and antenna gains.

    Args:
        d_bt: BS → Target distance (m)
        d_tr: Target → RX distance (m)
        beta: Target reflection coefficient (default 1.0)
        K0, alpha0, d0: Path loss parameters
        CAL: Calibration factor (default 1.0)

    Returns:
        alpha_sq: |alpha|^2 (scalar)
    """
    L_bt = path_loss_linear(d_bt, K0, alpha0, d0)
    L_tr = path_loss_linear(d_tr, K0, alpha0, d0)
    raw_alpha_sq = beta / (L_bt * L_tr)
    return raw_alpha_sq * CAL


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
        K0, alpha0, d0: Path loss parameters (CRB-Rate Tradeoff Eq.63)

    Returns:
        G: BS→IRS channel (N_irs × Mt)
    """
    L_br = path_loss_linear(d_br, K0, alpha0, d0)
    a_bs  = steering_vector(Mt, phi_br)     # Mt×1: BS steering toward IRS
    a_irs = steering_vector(N_irs, phi_br)   # N_irs×1: IRS steering from BS

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

    Rician fading (similar to BS→CU, CRB-Rate Tradeoff Eq.62).

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


def compute_effective_a(a, G, h_r, v, direct_blocked=False):
    """
    Compute effective target-direction steering vector  [new]

    a_eff(Θ) = a + (h_r @ Θ @ G)^T   (LoS: direct + IRS)
             = (h_r @ Θ @ G)^T         (NLoS: IRS only, direct blocked)

    where v = [e^{jθ₁}, ..., e^{jθ_N}]^T, Θ = diag(v)

    Args:
        a: Direct-path steering vector (Mt×1)
        G: BS→IRS channel (N_irs×Mt)
        h_r: IRS→Target channel (1×N_irs)
        v: IRS phase shift vector (N_irs,)
        direct_blocked: If True, remove direct BS→Target path

    Returns:
        a_eff: Effective steering vector (Mt×1)
    """
    # h_r @ Θ @ G → (1×N) @ (N×N) @ (N×Mt) = 1×Mt
    # h_r.T * v: element-wise → (N,)  (v[n] multiplies h_r[n])
    h_r_flat = h_r.flatten()  # (N,)
    irs_path = G.T @ (h_r_flat * v)  # Mt×N @ (N,) = (Mt,)
    if direct_blocked:
        # IRS-only path (no direct LoS)
        return irs_path.reshape(-1, 1)  # Mt×1
    # Both operands 1D → (Mt,), then reshape to column
    return (a.flatten() + irs_path).reshape(-1, 1)  # Mt×1


def irs_beam_align(h_r, G):
    """
    Align IRS phases to maximize signal toward target.

    Sets each element's phase to compensate the BS->IRS and IRS->Target paths,
    making all N reflected signals add coherently at the target.

    v_align[n] = exp(-j * (angle(G[n,0]) + angle(h_r[0,n])))

    Args:
        h_r: IRS->Target channel (1xN)
        G: BS->IRS channel (NxMt)

    Returns:
        v_align: Phase shift vector (N,) with |v| = 1
    """
    G_ref = G[:, 0]
    v = np.exp(-1j * (np.angle(h_r.flatten()) + np.angle(G_ref)))
    return v / (np.abs(v) + 1e-15)


def compute_effective_h(h, G, h_rc, v, direct_blocked=False):
    """
    Compute effective CU channel vector  [new]

    h_eff(Θ) = h + (h_rc @ Θ @ G)^T   (LoS: direct + IRS)
             = (h_rc @ Θ @ G)^T         (NLoS: IRS only, direct blocked)

    Args:
        h: Direct-path CU channel (Mt×1)
        G: BS→IRS channel (N_irs×Mt)
        h_rc: IRS→CU channel (1×N_irs)
        v: IRS phase shift vector (N_irs,)
        direct_blocked: If True, remove direct BS→CU path

    Returns:
        h_eff: Effective CU channel (Mt×1)
    """
    h_rc_flat = h_rc.flatten()  # (N,)
    irs_path = G.T @ (h_rc_flat * v)  # (Mt,)
    if direct_blocked:
        return irs_path.reshape(-1, 1)
    return (h.flatten() + irs_path).reshape(-1, 1)  # Mt×1
