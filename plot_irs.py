"""
IRS 结果可视化模块
================
IRS 项目的所有画图函数集中在此。

用法:
    from plot_irs import plot_irs_comparison
"""

import matplotlib.pyplot as plt
import numpy as np


LINE_STYLES = {
    "no_irs":   'k-',
    "nlos_irs": ['r--', 'b--', 'g--', 'm--', 'c--', 'y--'],
    "los_irs":  ['r:',  'b:',  'g:',  'm:',  'c:',  'y:'],
}


def plot_irs_comparison(data_no_irs, data_irs_list, labels=None,
                        save_path='irs_comparison.png', use_log=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Baseline: LoS (no IRS) solid black ----
    if data_no_irs is not None:
        s = np.argsort(data_no_irs['rate'])
        ax1.plot(data_no_irs['rate'][s], data_no_irs['crb'][s], 'k-',
                 linewidth=2.5, label='LoS (no IRS)', zorder=5)
        ax2.plot(10 * np.log10(data_no_irs['gamma'][s]), data_no_irs['crb'][s],
                 'k-', linewidth=2.5, label='LoS (no IRS)', zorder=5)

    # ---- NLoS+IRS curves: colored dashed lines ----
    nlos_idx = 0
    for i, d in enumerate(data_irs_list):
        if d is None or (labels and "LoS+IRS" in labels[i]):
            continue
        s = np.argsort(d['rate'])
        style = LINE_STYLES["nlos_irs"][nlos_idx % len(LINE_STYLES["nlos_irs"])]
        nlos_idx += 1
        lbl = labels[i] if labels else f'Config {i+1}'
        ax1.plot(d['rate'][s], d['crb'][s], style, linewidth=1.8, label=lbl)
        ax2.plot(10 * np.log10(d['gamma'][s]), d['crb'][s], style, linewidth=1.8, label=lbl)

    # ---- LoS+IRS: markers on solid line to show overlap with baseline ----
    for i, d in enumerate(data_irs_list):
        if d is None or not (labels and "LoS+IRS" in labels[i]):
            continue
        s = np.argsort(d['rate'])
        lbl = labels[i]
        for ax_ in [ax1, ax2]:
            x = d['rate'][s] if ax_ == ax1 else 10 * np.log10(d['gamma'][s])
            ax_.plot(x, d['crb'][s], 'c-s', markersize=3, markevery=5,
                     linewidth=1.2, label=lbl)

    ax1.set_xlabel('Communication Rate (bps/Hz)')
    ax1.set_ylabel('CRB for DoA Estimation (rad²)')
    ax1.set_title('IRS-ISAC: CRB vs Communication Rate')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc='best')
    if use_log:
        ax1.set_yscale('log')

    ax2.set_xlabel('SINR Threshold γ₀ (dB)')
    ax2.set_ylabel('CRB for DoA Estimation (rad²)')
    ax2.set_title('IRS-ISAC: CRB vs SINR Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc='best')
    if use_log:
        ax2.set_yscale('log')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}", flush=True)
    plt.close(fig)
