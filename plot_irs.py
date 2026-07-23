"""IRS 结果可视化"""
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['r', 'b', 'g', 'm', 'c', 'y']


def plot_irs_comparison(data_no_irs, data_irs_list, labels=None,
                        save_path='irs_comparison.png', use_log=False,
                        subtitle=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # LoS (no IRS) — 黑色实线
    if data_no_irs is not None:
        s = np.argsort(data_no_irs['rate'])
        ax1.plot(data_no_irs['rate'][s], data_no_irs['crb'][s], 'k-',
                 linewidth=2.5, label='LoS (no IRS)', zorder=5)
        ax2.plot(10 * np.log10(data_no_irs['gamma'][s]), data_no_irs['crb'][s],
                 'k-', linewidth=2.5, label='LoS (no IRS)', zorder=5)

    color_idx = 0
    for i, d in enumerate(data_irs_list):
        if d is None:
            continue
        lbl = labels[i] if labels else f'Config {i+1}'
        s = np.argsort(d['rate'])

        # 选颜色
        c = COLORS[color_idx % len(COLORS)]
        color_idx += 1

        # 选线型：Passive=虚线, Active=实线, LoS+IRS=点线+标记
        if "(P)" in lbl:
            style = c + '--'       # 虚线
        elif "(A)" in lbl:
            style = c + '-'        # 实线
        else:
            style = c + ':'        # 点线

        ax1.plot(d['rate'][s], d['crb'][s], style, linewidth=1.8, label=lbl)
        ax2.plot(10 * np.log10(d['gamma'][s]), d['crb'][s], style, linewidth=1.8, label=lbl)

    # 设置
    for ax_ in [ax1, ax2]:
        ax_.grid(alpha=0.3)
        ax_.legend(fontsize=7, loc='best')
        if use_log:
            ax_.set_yscale('log')

    ax1.set_xlabel('Communication Rate (bps/Hz)')
    ax1.set_ylabel('CRB (rad^2)')
    title_suffix = f"\n{subtitle}" if subtitle else ""
    ax1.set_title('IRS-ISAC: CRB vs Communication Rate' + title_suffix)

    ax2.set_xlabel('SINR Threshold (dB)')
    ax2.set_ylabel('CRB (rad^2)')
    ax2.set_title('IRS-ISAC: CRB vs SINR Threshold' + title_suffix)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}", flush=True)
    plt.close(fig)
