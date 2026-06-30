"""Replot combined comparison from saved NPZ data."""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plot_results import plot_irs_comparison

d = np.load(os.path.join(os.path.dirname(__file__), 'results/irs_comparison_all_20260630_153810.npz'))

los_no_irs = {'rate': d['LoS_no_IRS_rate'], 'crb': d['LoS_no_IRS_crb'], 'gamma': d['LoS_no_IRS_gamma']}
nlos_irs16 = {'rate': d['NLoS_IRS_N=16_rate'], 'crb': d['NLoS_IRS_N=16_crb'], 'gamma': d['NLoS_IRS_N=16_gamma']}
nlos_irs32 = {'rate': d['NLoS_IRS_N=32_rate'], 'crb': d['NLoS_IRS_N=32_crb'], 'gamma': d['NLoS_IRS_N=32_gamma']}
los_irs32  = {'rate': d['LoS_IRS_N=32_rate'], 'crb': d['LoS_IRS_N=32_crb'], 'gamma': d['LoS_IRS_N=32_gamma']}

plot_irs_comparison(
    los_no_irs,
    [nlos_irs16, nlos_irs32, los_irs32],
    labels=['NLoS, IRS N=16', 'NLoS, IRS N=32', 'LoS, IRS N=32'],
    save_path=os.path.join(os.path.dirname(__file__), 'results/comparison_all_combined.png'),
    use_log=True
)
print('Combined plot saved.')
