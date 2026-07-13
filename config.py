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
N_irs_list = [16, 32, 64, 128]  # IRS reflecting elements to scan
N_irs = 16                       # default value

# ===== Active IRS parameters (from Paper 7) =====
A_MAX = 8.0          # Maximum amplification gain (linear), ~18dB
P_RIS_dBm = 10.0     # RIS amplification power budget (dBm)
P_RIS = 10**((P_RIS_dBm - 30)/10)  # RIS power budget (linear, Watts)
SIGMA2_RIS_dBm = -80.0   # RIS amplification noise power (dBm)
SIGMA2_RIS = 10**((SIGMA2_RIS_dBm - 30)/10)

# ===== IRS positions (x, y) =====
# BS: (0, 0), Target: (200, 0), Sensing RX: (400, 0)
pos_bs     = np.array([0.0, 0.0])
pos_target = np.array([200.0, 0.0])
pos_rx     = np.array([400.0, 0.0])
pos_irs    = np.array([190.0, 5.0])     # IRS position (near target for NLoS)

# ===== Power & Noise =====
P_dBm      = 30.0
P          = 10**((P_dBm - 30)/10)   # 1 W
sigma2_c_dBm = -80.0
sigma2_s_dBm = -80.0
sigma2_c   = 10**((sigma2_c_dBm - 30)/10)  # CU noise power
sigma2_s   = 10**((sigma2_s_dBm - 30)/10)  # Sensing RX noise power

# ===== Geometry =====
theta_target = 0.0   # Target DoA (rad)
phi_target   = 0.0   # BS->Target direction (rad)
phi_cu       = 0.3   # BS->CU direction (rad)

# ===== Path loss (CRB-Rate Tradeoff Eq.63) =====
K0     = -30     # Path loss at d0 (dB)
alpha0 = 2.5     # Path loss exponent
d0     = 1.0     # Reference distance (m)

# ===== Target channel (CRB-Rate Tradeoff Eq.6) =====
CAL_ALPHA = 1.0e-32   # Calibration factor for |alpha|^2 (LoS scenario)
                        # NLoS+IRS 信号经 IRS 反射，前向路径已在 a_eff 中
                        # |alpha|² 只应含返回路径 L(Target→RX)

# ===== CU channel (CRB-Rate Tradeoff Eq.62) =====
Kc = 1.0        # Rician K-factor for BS->CU

# ===== SINR sweep =====
N_gamma        = 40     # SINR sweep points (-10dB ~ 19dB)
gamma_0_dB_min = -10.0
gamma_0_dB_max = 19.0

# ===== SDR solver (IRS) =====
SDR_TRIALS = 100    # Number of randomizations for SDR recovery

# ===== AO outer loop =====
AO_MAX_ITER = 20
AO_TOL      = 1e-4

# ===== Reproducibility =====
SEED       = 0      # CRB-Rate Tradeoff uses h_seed=46 for channel gen
SEED_CHANNEL = 46   # Same seed for fair comparison
