"""
场景共用信道参数初始化
======================
一次性算好所有场景共享的几何量（距离、角度）和信道量（信道矩阵），
后面各场景复用，不随 SINR 扫描点变化。

用法:
    from channel_constant import init_geometry_and_channels, generate_irs_channels
    geo, ch = init_geometry_and_channels()
    irs_ch = generate_irs_channels(N_irs=16, irs_pos=pos_irs, geo=geo)
"""

import numpy as np
from steering_vectors import steering_vector, steering_vector_derivative
from config import (
    Mt, Mr, Kc, K0, alpha0, d0, CAL_ALPHA, SEED_CHANNEL,
    pos_bs, pos_target, pos_rx, pos_irs,
    phi_target, theta_target, phi_cu,
)
from channels import (
    compute_distance, compute_angle,
    generate_rician_channel, compute_alpha_sq,
    generate_irs_bs_channel, generate_irs_target_channel, generate_irs_cu_channel,
)


def init_geometry_and_channels():
    """
    Compute all shared geometry (distances/angles) and channels once.
    Called once at startup; results reused across all SINR sweep points.

    Returns:
        geo: dict with distances and angles (fixed: BS/Target/RX/CU)
        ch:  dict with channel matrices and steering vectors
    """
    d_bt = compute_distance(pos_bs, pos_target)
    d_tr = compute_distance(pos_target, pos_rx)
    d_bc = compute_distance(pos_bs, np.array([1000.0 * np.cos(phi_cu), 1000.0 * np.sin(phi_cu)]))

    h = generate_rician_channel(Mt, phi_cu, Kc, d_bc, K0, alpha0, d0, SEED_CHANNEL)
    a_dir = steering_vector(Mt, phi_target)
    b = steering_vector(Mr, theta_target)
    b_dot = steering_vector_derivative(Mr, theta_target)
    alpha_sq = compute_alpha_sq(d_bt, d_tr, 1.0, K0, alpha0, d0, CAL_ALPHA)

    geo = {
        "d_bt": d_bt, "d_tr": d_tr, "d_bc": d_bc,
        "pos_cu": np.array([1000.0 * np.cos(phi_cu), 1000.0 * np.sin(phi_cu)]),
    }
    ch = {
        "h": h, "a_dir": a_dir, "b": b, "b_dot": b_dot,
        "alpha_sq": alpha_sq,
    }
    return geo, ch


def compute_irs_geometry(irs_pos, geo):
    """
    Compute distances and angles for a specific IRS position.

    Args:
        irs_pos: (x, y) array of IRS position
        geo: shared geometry dict (needs pos_cu)

    Returns:
        dict with d_br, d_rt, d_rc, phi_br, phi_rt, phi_rc
    """
    d_br = compute_distance(pos_bs, irs_pos)
    d_rt = compute_distance(irs_pos, pos_target)
    d_rc = compute_distance(irs_pos, geo["pos_cu"])
    phi_br = compute_angle(pos_bs, irs_pos)
    phi_rt = compute_angle(irs_pos, pos_target)
    phi_rc = compute_angle(irs_pos, geo["pos_cu"])
    return {"d_br": d_br, "d_rt": d_rt, "d_rc": d_rc,
            "phi_br": phi_br, "phi_rt": phi_rt, "phi_rc": phi_rc}


def generate_irs_channels(N_irs, irs_pos, geo):
    """
    Generate IRS channels (G, h_r, h_rc) for a specific IRS position and size.

    Args:
        N_irs: number of IRS elements
        irs_pos: (x, y) position of IRS
        geo: shared geometry dict

    Returns:
        dict with "G" (N_irs×Mt), "h_r" (1×N_irs), "h_rc" (1×N_irs)
    """
    irs_geo = compute_irs_geometry(irs_pos, geo)
    G = generate_irs_bs_channel(Mt, N_irs, irs_geo["d_br"], irs_geo["phi_br"], K0, alpha0, d0)
    h_r = generate_irs_target_channel(N_irs, irs_geo["d_rt"], irs_geo["phi_rt"], K0, alpha0, d0)
    h_rc = generate_irs_cu_channel(N_irs, irs_geo["d_rc"], irs_geo["phi_rc"], Kc, K0, alpha0, d0, SEED_CHANNEL)
    return {"G": G, "h_r": h_r, "h_rc": h_rc}
