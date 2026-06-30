"""
IRS 结果可视化模块
================
IRS 项目的所有画图函数集中在此。

用法:
    from plot_irs import plot_irs_comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_irs_comparison(data_no_irs, data_irs_list, labels=None,
                        save_path='irs_comparison.png', use_log=False):
    """
    Compare CRB-Rate tradeoff with and without IRS.

    Args:
        data_no_irs: dict with 'rate', 'crb', 'gamma' (baseline, no IRS)
        data_irs_list: list of dicts, each with 'rate', 'crb', 'gamma'
        labels: list of legend labels for each IRS config
        save_path: output path
        use_log: if True, use log scale for CRB y-axis
    """
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
        if d is None:
            continue
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
    if use_log:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    else:
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.set_xlabel('SINR Threshold γ₀ (dB)')
    ax2.set_ylabel('CRB for DoA Estimation (rad²)')
    ax2.set_title('CRB vs SINR Constraint')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    if not use_log:
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show(block=False)
    plt.pause(0.5)
