"""Channel, single-pass IRS noise, and IRS output-power models."""

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
    # A local legacy generator preserves the original seeded realization
    # without resetting NumPy's process-wide random state.
    rng = np.random.RandomState(seed)

    # LoS component: steering vector toward CU direction
    h_los = steering_vector(Mt, phi_cu).flatten()

    # NLoS component: Rayleigh fading
    h_nlos = (rng.randn(Mt) + 1j * rng.randn(Mt)) / np.sqrt(2)

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

    |alpha|^2 = |beta|^2 L(d_BT) L(d_TR) * CAL

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
    raw_alpha_sq = abs(beta) ** 2 * L_bt * L_tr
    return raw_alpha_sq * CAL


def compute_return_alpha_sq(d_tr, beta=1.0,
                            K0=-30, alpha0=2.5, d0=1.0, CAL=1.0):
    """
    Return-path coefficient when forward path loss is inside a_eff.

    We factor the bistatic channel as

        H = alpha_return * b * a_forward^T,

    where a_forward already contains BS→Target (or BS→IRS→Target)
    attenuation. Then |alpha_return|² = |beta|² L(Target→RX).
    For a direct path this is algebraically equivalent to the full cascade
    returned by compute_alpha_sq().
    """
    L_tr = path_loss_linear(d_tr, K0, alpha0, d0)
    return abs(beta) ** 2 * L_tr * CAL


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
    Generate the IRS → Target column channel h_r ∈ ℂ^{N_irs×1}.

    IRS reflects signal toward target. The cascaded path is
    BS→IRS→Target→SensingRX. The h_r is the IRS→Target steering.

    Args:
        N_irs: IRS elements
        d_rt: IRS-Target distance via target reflection (m)
        phi_rt: Target direction from IRS (rad)

    Returns:
        h_r: IRS→Target column channel (N_irs × 1)
    """
    L_rt = path_loss_linear(d_rt, K0, alpha0, d0)
    a_irs_target = steering_vector(N_irs, phi_rt)  # N_irs×1

    # Store receiver channels as columns so received = h^H x everywhere.
    h_r = np.sqrt(L_rt) * a_irs_target
    return h_r


def generate_irs_cu_channel(N_irs, d_rc, phi_rc, Kc=1.0,
                            K0=-30, alpha0=2.5, d0=1.0, seed=42):
    """
    Generate the IRS → CU column channel h_rc ∈ ℂ^{N_irs×1}.

    Rician fading (similar to BS→CU, CRB-Rate Tradeoff Eq.62).

    Channel convention used throughout the project:

        received scalar = h^H x

    Therefore every single-antenna receiver channel is stored as a column
    vector. Keeping h_rc as a column makes the cascaded BS→IRS→CU channel
    unambiguous:

        h_irs = G^H Φ^H h_rc  ∈ ℂ^{Mt×1}.

    Args:
        N_irs: IRS elements
        d_rc: IRS-CU distance (m)
        phi_rc: CU direction from IRS (rad)
        Kc: Rician K-factor
        seed: Random seed
    """
    rng = np.random.RandomState(seed)
    L_rc = path_loss_linear(d_rc, K0, alpha0, d0)

    # LoS component
    h_los = steering_vector(N_irs, phi_rc).flatten()  # N_irs,

    # NLoS component
    h_nlos = (
        rng.randn(N_irs) + 1j * rng.randn(N_irs)
    ) / np.sqrt(2)

    # Rician combination
    h_channel = (np.sqrt(Kc / (Kc + 1)) * h_los
                 + np.sqrt(1 / (Kc + 1)) * h_nlos)

    return np.sqrt(L_rc) * h_channel.reshape(-1, 1)  # N_irs×1


def compute_effective_a(a, G, h_r, v, direct_blocked=False):
    """
    Compute the transmit steering vector used by
    CRB-Rate Tradeoff for Bistatic ISAC, including the IRS path.

    CRB-Rate Tradeoff for Bistatic ISAC defines H = alpha b a^T, so

        r_t = (a^T + h_r^T Φ G) x,

    and therefore

        a_eff = a + G^T Φ h_r.

    This target convention intentionally differs from the communication
    convention y_c = h_eff^H x.
    """
    a_direct = np.asarray(a).reshape(-1, 1)
    h_r_col = np.asarray(h_r).reshape(-1, 1)
    v_col = np.asarray(v).reshape(-1, 1)
    if G.shape[0] != h_r_col.shape[0] or h_r_col.shape != v_col.shape:
        raise ValueError("Incompatible IRS dimensions for target channel.")

    # G^T Φ h_r = G^T (v ⊙ h_r)
    irs_path = G.T @ (v_col * h_r_col)
    if direct_blocked:
        return irs_path
    return a_direct + irs_path


def irs_beam_align(h_r, G):
    """
    Align IRS phases to maximize signal toward target.

    Sets each element's phase to compensate the BS->IRS and IRS->Target paths,
    making all N reflected signals add coherently at the target.

    Under the target convention in CRB-Rate Tradeoff for Bistatic ISAC,

        a_eff = G^T diag(v) h_r.

    For the rank-one LoS G used in this project, using the first BS
    antenna as phase reference gives

        angle(v_n) = -angle(h_r,n) - angle(G_n,0).

    Args:
        h_r: IRS->Target column channel (Nx1)
        G: BS->IRS channel (NxMt)

    Returns:
        v_align: Phase shift vector (N,) with |v| = 1
    """
    G_ref = G[:, 0]
    h_r_flat = np.asarray(h_r).reshape(-1)
    v = np.exp(-1j * (np.angle(h_r_flat) + np.angle(G_ref)))
    return v / (np.abs(v) + 1e-15)


def compute_effective_h(h, G, h_rc, v, direct_blocked=False):
    """
    Compute the effective BS → CU column channel.

    Start from the physical received-signal expression

        y_c = (h^H + h_rc^H Φ G) x + noise.

    To retain the standard column-channel form y_c = h_eff^H x + noise,
    take the Hermitian transpose of the reflected row channel:

        h_eff = h + G^H Φ^H h_rc.

    Since Φ = diag(v), multiplying Φ^H h_rc is just the element-wise
    product conj(v) * h_rc. We do not build the full diagonal matrix.

    Args:
        h: Direct-path CU channel (Mt×1)
        G: BS→IRS channel (N_irs×Mt)
        h_rc: IRS→CU column channel (N_irs×1)
        v: IRS reflection coefficients (N_irs,)
        direct_blocked: If True, remove direct BS→CU path

    Returns:
        h_eff: Effective CU channel (Mt×1)
    """
    h_direct = np.asarray(h).reshape(-1, 1)
    h_rc_col = np.asarray(h_rc).reshape(-1, 1)
    v_col = np.asarray(v).reshape(-1, 1)

    if G.shape[0] != h_rc_col.shape[0] or h_rc_col.shape != v_col.shape:
        raise ValueError(
            "Incompatible IRS dimensions: expected G=(N_irs,Mt), "
            "h_rc=(N_irs,1), and v=(N_irs,)."
        )

    # G^H Φ^H h_rc = G^H (conj(v) ⊙ h_rc)
    irs_path = G.conj().T @ (v_col.conj() * h_rc_col)

    if direct_blocked:
        return irs_path
    return h_direct + irs_path


def compute_active_irs_noise_power(h_rc, v, sigma2_irs):
    """
    Compute Active-IRS amplifier noise power observed by the CU.

    The IRS creates z_I ~ CN(0, sigma2_irs I). After reflection, the
    noise arriving at the CU is

        n_irs = h_rc^H Φ z_I.

    Its variance is

        E[|n_irs|²] = sigma2_irs ||h_rc^H Φ||²
                    = sigma2_irs ||Φ^H h_rc||².

    This is noise power, not part of h_eff: h_eff multiplies the BS
    signal x, whereas this term multiplies independent IRS noise.
    """
    if sigma2_irs < 0:
        raise ValueError("sigma2_irs must be non-negative.")

    h_rc_col = np.asarray(h_rc).reshape(-1, 1)
    v_col = np.asarray(v).reshape(-1, 1)
    if h_rc_col.shape != v_col.shape:
        raise ValueError("h_rc and v must contain the same number of IRS elements.")

    phi_h_h_rc = v_col.conj() * h_rc_col
    return float(sigma2_irs * np.linalg.norm(phi_h_h_rc) ** 2)


def compute_active_irs_output_power(G, Rc, Rs, v, sigma2_irs):
    """Compute exact Active-IRS output power after solving Rc and Rs."""
    G = np.asarray(G)
    Rc = np.asarray(Rc)
    Rs = np.asarray(Rs)
    v_col = np.asarray(v).reshape(-1)
    if G.ndim != 2 or G.shape[0] != v_col.size:
        raise ValueError("G rows and v length must equal the number of IRS elements.")
    if Rc.shape != Rs.shape or Rc.shape != (G.shape[1], G.shape[1]):
        raise ValueError("Rc and Rs must match the BS-side dimension of G.")
    if sigma2_irs < 0:
        raise ValueError("sigma2_irs must be non-negative.")

    # P_out = sum_n |v_n|² ([G(Rc+Rs)G^H]_{n,n} + sigma_I²).
    incident_signal = np.real(np.diag(G @ (Rc + Rs) @ G.conj().T))
    per_element_input = np.maximum(incident_signal, 0.0) + sigma2_irs
    return float(np.sum(np.abs(v_col) ** 2 * per_element_input))


def compute_forwarded_irs_sensing_noise_power(
    h_r, v, b, alpha_sq, sigma2_irs
):
    """
    Total sensing-array power of first-pass IRS noise reflected by the target.

    In this project's one-pass bistatic topology the neglected term is

        n_fwd = alpha_return b h_r^T Phi z_I.

    Its covariance is rank one and its total power is

        |alpha_return|² sigma_I² ||h_r^T Phi||² ||b||².

    Cramér-Rao Bound Optimization for Active RIS-Empowered ISAC Systems
    neglects the analogous first-pass term after multiple path-loss factors.
    We compute it as a diagnostic but do not insert it into the white-noise
    CRB, which would require a newly derived FIM.
    """
    if min(alpha_sq, sigma2_irs) < 0:
        raise ValueError("alpha_sq and sigma2_irs must be non-negative.")
    h_r_col = np.asarray(h_r).reshape(-1)
    v_col = np.asarray(v).reshape(-1)
    b_col = np.asarray(b).reshape(-1)
    if h_r_col.shape != v_col.shape:
        raise ValueError("h_r and v must contain the same number of elements.")
    scalar_variance = sigma2_irs * np.sum(np.abs(h_r_col * v_col) ** 2)
    return float(alpha_sq * scalar_variance * np.linalg.norm(b_col) ** 2)


def compute_safe_uniform_active_gain(G, bs_power, irs_power, sigma2_irs,
                                     amplitude_limit):
    """
    Largest uniform Active-IRS gain guaranteed to obey its output-power cap.

    Since tr(R_x) <= bs_power implies
    [G R_x G^H]_(n,n) <= bs_power ||G[n,:]||², this conservative bound is
    valid before the BS covariance is known. The AO/SDR solver instead uses
    the exact covariance-dependent IRS power constraint.
    """
    if min(bs_power, irs_power, sigma2_irs, amplitude_limit) < 0:
        raise ValueError("Power, noise, and amplitude limits must be non-negative.")

    worst_case_input = (
        bs_power * np.sum(np.abs(G) ** 2, axis=1) + sigma2_irs
    )
    denominator = float(np.sum(worst_case_input))
    if denominator <= 0:
        return float(amplitude_limit)
    return float(min(amplitude_limit, np.sqrt(irs_power / denominator)))
