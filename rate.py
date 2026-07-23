"""Communication SINR and rate formulas."""

import numpy as np


def compute_rate_case1(Rc, h, sigma2_c):
    """
    Compute communication SINR and rate at CU  [Eq.(4)].
    Case 1: Gaussian signals only (no interference).

    gamma_1 = h^H * R_c * h / sigma2_c
    R_1     = log2(1 + gamma_1)

    Args:
        Rc: Transmit covariance matrix (Mt x Mt)
        h: BS -> CU channel vector (Mt, 1)
        sigma2_c: CU noise power (scalar)

    Returns:
        rate: Achievable rate (bps/Hz)
        sinr: SINR value (linear scale)
    """
    sinr = (h.conj().T @ Rc @ h).real / sigma2_c
    rate = np.log2(1 + max(sinr, 0))
    return rate, sinr


def compute_rate_case2(Rc, Rs, h, sigma2_c):
    """
    Compute communication SINR and rate at CU  [Eq.(13)-(14)].
    Case 2: Superposition of Gaussian + deterministic signals.

    gamma_2 = h^H * R_c * h / (h^H * R_s * h + sigma2_c)
    R_2     = log2(1 + gamma_2)

    Args:
        Rc: Information signal covariance matrix (Mt x Mt)
        Rs: Deterministic sensing signal covariance matrix (Mt x Mt)
        h: BS -> CU channel vector (Mt, 1)
        sigma2_c: CU noise power (scalar)

    Returns:
        rate: Achievable rate (bps/Hz)
        sinr: SINR value (linear scale)
    """
    signal   = (h.conj().T @ Rc @ h).real
    interfer = (h.conj().T @ Rs @ h).real
    sinr = signal / (interfer + sigma2_c)
    rate = np.log2(1 + max(sinr, 0))
    return rate, sinr


def compute_rate_irs(Rc, Rs, h_eff, sigma2_c, irs_noise_power=0.0):
    """
    Compute the CU rate with an IRS-enhanced effective channel.

    For a Passive IRS, irs_noise_power is zero and this reduces to Case 2 in
    CRB-Rate Tradeoff for Bistatic ISAC. An Active IRS has amplifiers at its
    elements, so their independent noise is added to the denominator:

        SINR = h_eff^H Rc h_eff
               / (h_eff^H Rs h_eff + sigma2_c + irs_noise_power).

    Args:
        Rc: Information covariance (Mt×Mt)
        Rs: Sensing covariance (Mt×Mt)
        h_eff: Effective CU channel (Mt×1)
        sigma2_c: CU noise power
        irs_noise_power: Active-IRS noise power after the IRS→CU channel

    Returns:
        rate: Achievable rate (bps/Hz)
        sinr: SINR (linear scale)
    """
    if irs_noise_power < 0:
        raise ValueError("irs_noise_power must be non-negative.")

    signal = float(np.real(h_eff.conj().T @ Rc @ h_eff).item())
    sensing_interference = float(np.real(h_eff.conj().T @ Rs @ h_eff).item())
    total_noise = sigma2_c + irs_noise_power

    sinr = signal / (sensing_interference + total_noise)
    rate = np.log2(1 + max(sinr, 0.0))
    return float(rate), float(sinr)
