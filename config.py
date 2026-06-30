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
N_irs = 16         # IRS reflecting elements (start at 16, can go to 32)

# ===== IRS positions (x, y) =====
# BS: (0, 0), Target: (200, 0), Sensing RX: (400, 0)
pos_bs     = np.array([0.0, 0.0])
pos_target = np.array([200.0, 0.0])
pos_rx     = np.array([400.0, 0.0])
pos_irs    = np.array([50.0, 30.0])   # typical IRS placement near BS

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

# ===== Path loss (Paper 4 Eq.63) =====
K0     = -30     # Path loss at d0 (dB)
alpha0 = 2.5     # Path loss exponent
d0     = 1.0     # Reference distance (m)

# ===== Target channel (Paper 4 Eq.6) =====
CAL_ALPHA = 1.0e-32   # Calibration factor for |alpha|^2

# ===== CU channel (Paper 4 Eq.62) =====
Kc = 1.0        # Rician K-factor for BS->CU

# ===== SINR sweep =====
N_gamma        = 40
gamma_0_dB_min = -10.0
gamma_0_dB_max = 19.0

# ===== SDR solver (IRS) =====
SDR_TRIALS = 100    # Number of randomizations for SDR recovery

# ===== AO outer loop =====
AO_MAX_ITER = 20
AO_TOL      = 1e-4

# ===== Reproducibility =====
SEED       = 0      # Paper 4 uses h_seed=46 for channel gen
SEED_CHANNEL = 46   # Keep Paper 4's channel seed for fair comparison
