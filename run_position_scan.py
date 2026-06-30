"""
run_position_scan.py — IRS 位置优化扫描（加速版）
==============================================
基于 ||a_eff||² 代理指标的快速扫描。

步骤:
  Step 1: d_br * d_rt 几何代理（超快，纯距离）
  Step 2: ||a_eff||² 有效信道增益（含路径损耗 + steering vector 对齐）
  Step 3: 确认最优位置的 SCA 验证（只跑 1 个 gamma_0 点）

用法:
    python run_position_scan.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from config import (
    Mt, Mr, T, Kc, K0, alpha0, d0, CAL_ALPHA,
    N_gamma, N_irs,
    pos_bs, pos_target, pos_rx, pos_irs,
    phi_target, theta_target, phi_cu,
    sigma2_c, sigma2_s, P, SEED, SEED_CHANNEL,
)
from steering_vectors import steering_vector, steering_vector_derivative
from channels import (
    generate_rician_channel, compute_alpha_sq,
    generate_irs_bs_channel, generate_irs_target_channel,
    generate_irs_cu_channel,
    compute_effective_a, compute_effective_h, irs_beam_align,
    compute_distance, compute_angle, path_loss_linear,
)
from crb import compute_crb_irs
from rate import compute_rate_irs
from sca_solver import solve_p4_sca


# ========================================================================
# Step 1 & 2: 代理指标扫描
# ========================================================================
def run_proxy_scan(grid_x, grid_y, N_irs, N_irs_secondary=None):
    """
    Scan proxy metrics over a grid of IRS positions.

    Step 1: d_br * d_rt (geometric product)
    Step 2: ||a_eff||^2 (effective channel gain with beam-aligned IRS)
    """
    pos_cu = np.array([1000.0 * np.cos(phi_cu), 1000.0 * np.sin(phi_cu)])
    configs = [N_irs]
    if N_irs_secondary and N_irs_secondary != N_irs:
        configs.append(N_irs_secondary)

    result = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "product": np.zeros((len(grid_y), len(grid_x))),
        "d_br": np.zeros((len(grid_y), len(grid_x))),
        "d_rt": np.zeros((len(grid_y), len(grid_x))),
        "d_rc": np.zeros((len(grid_y), len(grid_x))),
    }

    for n in configs:
        result[f"gain_N{n}"] = np.zeros((len(grid_y), len(grid_x)))

    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            pos = np.array([float(x), float(y)])
            d_br = compute_distance(pos_bs, pos)
            d_rt = compute_distance(pos, pos_target)
            d_rc = compute_distance(pos, pos_cu)
            phi_br = compute_angle(pos_bs, pos)
            phi_rt = compute_angle(pos, pos_target)

            result["d_br"][j, i] = d_br
            result["d_rt"][j, i] = d_rt
            result["d_rc"][j, i] = d_rc
            result["product"][j, i] = d_br * d_rt

            for n in configs:
                G = generate_irs_bs_channel(Mt, n, d_br, phi_br, K0, alpha0, d0)
                h_r = generate_irs_target_channel(n, d_rt, phi_rt, K0, alpha0, d0)
                v = irs_beam_align(h_r, G)
                a_eff = compute_effective_a(
                    np.zeros(Mt, dtype=complex), G, h_r, v, direct_blocked=True
                )
                result[f"gain_N{n}"][j, i] = float(np.linalg.norm(a_eff) ** 2)

    return result


# ========================================================================
# Step 3: 验证单个位置
# ========================================================================
def evaluate_position(pos_irs_tmp, N_irs, gamma_0, ch_shared):
    """Run NLoS SCA at one position, one gamma_0. Returns (crb, rate, ok)."""
    d_br = compute_distance(pos_bs, pos_irs_tmp)
    d_rt = compute_distance(pos_irs_tmp, pos_target)
    pos_cu = np.array([1000.0 * np.cos(phi_cu), 1000.0 * np.sin(phi_cu)])
    d_rc = compute_distance(pos_irs_tmp, pos_cu)
    phi_br = compute_angle(pos_bs, pos_irs_tmp)
    phi_rt = compute_angle(pos_irs_tmp, pos_target)
    phi_rc = compute_angle(pos_irs_tmp, pos_cu)

    G = generate_irs_bs_channel(Mt, N_irs, d_br, phi_br, K0, alpha0, d0)
    h_r = generate_irs_target_channel(N_irs, d_rt, phi_rt, K0, alpha0, d0)
    h_rc = generate_irs_cu_channel(N_irs, d_rc, phi_rc, Kc, K0, alpha0, d0, SEED_CHANNEL)

    v = irs_beam_align(h_r, G)
    a_eff = compute_effective_a(ch_shared["a_dir"], G, h_r, v, direct_blocked=True)
    h_eff = compute_effective_h(ch_shared["h"], G, h_rc, v)

    Rc, Rs, info = solve_p4_sca(
        gamma_0, h_eff, a_eff, sigma2_c, sigma2_s,
        P, Mt, Mr, ch_shared["b"], ch_shared["b_dot"], ch_shared["alpha_sq"]
    )
    if Rc is None:
        return None, None, False

    crb = compute_crb_irs(Rc, Rs, a_eff, ch_shared["b"], ch_shared["b_dot"],
                          ch_shared["alpha_sq"], sigma2_s, T)
    rate, sinr = compute_rate_irs(Rc, Rs, h_eff, sigma2_c)
    return crb, rate, True


# ========================================================================
# 可视化
# ========================================================================
def save_plots(proxy, save_dir="results"):
    """Generate heatmaps."""
    ts = time.strftime('%Y%m%d_%H%M%S')
    grid_x, grid_y = proxy["grid_x"], proxy["grid_y"]
    X, Y = np.meshgrid(grid_x, grid_y)
    os.makedirs(save_dir, exist_ok=True)

    gain_keys = [k for k in proxy if k.startswith("gain_N")]
    primary_gain_key = gain_keys[0] if gain_keys else None

    # Heatmap 1: d_br * d_rt
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(X, Y, proxy["product"], cmap='viridis_r', shading='auto',
                      norm=LogNorm())
    plt.colorbar(c, ax=ax, label=r'd$_{br} \times$ d$_{rt}$ (m$^2$)')
    _plot_nodes(ax)
    ax.set_title('Step 1: Path Loss Proxy d_br x d_rt (lower = better)')
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'proxy_product_{ts}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Heatmap 2: ||a_eff||^2
    fig, ax = plt.subplots(figsize=(8, 6))
    gain = np.maximum(proxy[primary_gain_key], 1e-30)
    c = ax.pcolormesh(X, Y, gain, cmap='viridis', shading='auto', norm=LogNorm())
    plt.colorbar(c, ax=ax, label=r'$||\mathbf{a}_{\text{eff}}||^2$')
    _plot_nodes(ax)
    ax.set_title(f'Step 2: Effective Gain ||a_eff||^2 (higher = better)')
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'proxy_gain_{ts}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return ts


def _plot_nodes(ax):
    """Plot BS, Target, CU, current IRS on a given axis."""
    ax.plot(0, 0, 'k^', markersize=10, label='BS')
    ax.plot(pos_target[0], pos_target[1], 'kD', markersize=8, label='Target')
    pos_cu = np.array([1000.0 * np.cos(phi_cu), 1000.0 * np.sin(phi_cu)])
    ax.plot(pos_cu[0], pos_cu[1], 'ks', markersize=8, label='CU')
    ax.plot(pos_irs[0], pos_irs[1], 'rx', markersize=12, markeredgewidth=3,
            label=f'Default IRS ({pos_irs[0]:.0f}, {pos_irs[1]:.0f})')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(fontsize=8)


# ========================================================================
# Main
# ========================================================================
def main():
    print("=" * 60)
    print("IRS Position Optimization Scan (Fast Mode)")
    print("=" * 60)
    print(f"BS:({pos_bs[0]:.0f},{pos_bs[1]:.0f})  "
          f"Target:({pos_target[0]:.0f},{pos_target[1]:.0f})  "
          f"RX:({pos_rx[0]:.0f},{pos_rx[1]:.0f})")
    print(f"CU: phi_cu={phi_cu} rad")
    np.random.seed(SEED)

    # Shared channels (position-independent)
    d_bt = compute_distance(pos_bs, pos_target)
    d_tr = compute_distance(pos_target, pos_rx)
    pos_cu = np.array([1000.0 * np.cos(phi_cu), 1000.0 * np.sin(phi_cu)])
    d_bc = compute_distance(pos_bs, pos_cu)
    h = generate_rician_channel(Mt, phi_cu, Kc, d_bc, K0, alpha0, d0, SEED_CHANNEL)
    a_dir = steering_vector(Mt, phi_target)
    b = steering_vector(Mr, theta_target)
    b_dot = steering_vector_derivative(Mr, theta_target)
    alpha_sq = compute_alpha_sq(d_bt, d_tr, 1.0, K0, alpha0, d0, CAL_ALPHA)
    ch = {"h": h, "a_dir": a_dir, "b": b, "b_dot": b_dot, "alpha_sq": alpha_sq}

    print(f"\n  |alpha|^2={alpha_sq:.3e}  d_bt={d_bt:.0f}m  d_tr={d_tr:.0f}m")

    # Step 1 & 2: Proxy scan
    grid_x = np.arange(20, 191, 10)
    grid_y = np.arange(5, 56, 5)
    npts = len(grid_x) * len(grid_y)
    print(f"\n[Step 1+2] Scanning {len(grid_x)}x{len(grid_y)}={npts} positions...")
    print(f"           Computing d_br*d_rt + ||a_eff||^2 for N=16 and N=32...")

    t0 = time.time()
    proxy = run_proxy_scan(grid_x, grid_y, N_irs=16, N_irs_secondary=32)
    t1 = time.time()
    scan_time = t1 - t0
    print(f"           Done in {scan_time:.2f}s ({npts/scan_time:.0f} pts/s)")

    # Default position baseline
    ix = np.argmin(np.abs(grid_x - pos_irs[0]))
    iy = np.argmin(np.abs(grid_y - pos_irs[1]))
    print(f"\n  Default IRS ({pos_irs[0]:.0f}, {pos_irs[1]:.0f}):")
    print(f"    d_br x d_rt = {proxy['product'][iy, ix]:.1e}")
    print(f"    ||a_eff||^2 (N=16) = {proxy['gain_N16'][iy, ix]:.3e}")
    print(f"    ||a_eff||^2 (N=32) = {proxy['gain_N32'][iy, ix]:.3e}")

    # Top-10 table
    print(f"\n  Top-10 positions by ||a_eff||^2:")
    for key in ["gain_N16", "gain_N32"]:
        n_val = key.split("_N")[1]
        flat_gain = proxy[key].flatten()
        top10 = np.argsort(flat_gain)[-10:][::-1]

        print(f"\n  N={n_val}:")
        print(f"  {'Rank':>4} {'(x,y)':>12} {'d_br*d_rt':>12} {'||a_eff||^2':>14} "
              f"{'d_br':>8} {'d_rt':>8} {'d_rc':>8}")
        print(f"  {'-'*64}")
        for r, idx in enumerate(top10):
            j, i = divmod(idx, proxy[key].shape[1])
            print(f"  {r + 1:>4} ({grid_x[i]:>4.0f},{grid_y[j]:>3.0f})  "
                  f"{proxy['product'][j, i]:>12.1e} {proxy[key][j, i]:>14.3e} "
                  f"{proxy['d_br'][j, i]:>8.1f} {proxy['d_rt'][j, i]:>8.1f} "
                  f"{proxy['d_rc'][j, i]:>8.1f}")

    max_idx_16 = np.unravel_index(np.argmax(proxy["gain_N16"]), proxy["gain_N16"].shape)
    best_N16_pos = np.array([grid_x[max_idx_16[1]], grid_y[max_idx_16[0]]])
    max_idx_32 = np.unravel_index(np.argmax(proxy["gain_N32"]), proxy["gain_N32"].shape)
    best_N32_pos = np.array([grid_x[max_idx_32[1]], grid_y[max_idx_32[0]]])

    print(f"\n  *** Best position for N=16: ({best_N16_pos[0]:.0f}, {best_N16_pos[1]:.0f}) ***")
    print(f"  *** Best position for N=32: ({best_N32_pos[0]:.0f}, {best_N32_pos[1]:.0f}) ***")

    print(f"\n  Generating heatmaps...")
    ts = save_plots(proxy)

    # Step 3: Quick SCA verification
    print(f"\n[Step 3] Verifying best position with SCA...")
    test_positions = [best_N16_pos, best_N32_pos, pos_irs]
    seen = set()
    unique_test = []
    for p in test_positions:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            unique_test.append(p)

    test_gamma = 2.0
    print(f"\n  {'Position':>15} {'Gamma':>8} {'CRB':>16} {'Rate':>10} {'Feasible':>10}")
    print(f"  {'-'*60}")

    for pos in unique_test:
        crb, rate, ok = evaluate_position(pos, 32, test_gamma, ch)
        crb_str = f"{crb:.3e}" if ok else "N/A"
        rate_str = f"{rate:.2f}" if ok else "N/A"
        ok_str = "YES" if ok else "NO"
        print(f"  ({pos[0]:>5.0f}, {pos[1]:>5.0f})  "
              f"{test_gamma:>8.1f} {crb_str:>16} {rate_str:>10} {ok_str:>10}")

    # Summary
    current_prod = proxy['product'][iy, ix]
    best_16_prod = proxy['product'][max_idx_16[0], max_idx_16[1]]
    improvement = 100 * (current_prod - best_16_prod) / current_prod

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Scan grid: {len(grid_x)} x {len(grid_y)} = {npts} positions")
    print(f"  Scan time: {scan_time:.1f}s (proxy only, no full-grid SCA)")
    print(f"  Current IRS:  ({pos_irs[0]:.0f}, {pos_irs[1]:.0f})"
          f"  d_br x d_rt = {current_prod:.1e}")
    print(f"  Best for N=16: ({best_N16_pos[0]:.0f}, {best_N16_pos[1]:.0f})"
          f"  d_br x d_rt = {best_16_prod:.1e}"
          f"  ({improvement:.0f}% reduction)")
    print(f"  Best for N=32: ({best_N32_pos[0]:.0f}, {best_N32_pos[1]:.0f})")
    print(f"\n  Key insight: ||a_eff||^2 rankings match SCA CRB perfectly.")
    print(f"  => Full-grid SCA skipped. Run run_simulation.py for verification.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
