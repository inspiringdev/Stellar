"""
STELLAR Configuration — matches simulation parameters from the paper.
Section IV: Simulation Results
"""

import numpy as np

# ─────────────────────────────────────────────
# Satellite / Channel Parameters
# ─────────────────────────────────────────────
Nt_DEFAULT   = 16        # transmit antennas at SBS
K_DEFAULT    = 4         # number of ground stations
Pt_dBm       = 33        # total transmit power (dBm)
Pt           = 10 ** ((Pt_dBm - 30) / 10)   # Watts ≈ 2 W
ALTITUDE_KM  = 780       # LEO altitude (km)
FC_GHz       = 4.0       # carrier frequency (GHz)
GS_dBi       = 35.0      # satellite antenna gain (dBi)
GK_dBi       = 24.3      # GS antenna gain (dBi)
NOISE_dBm    = -90.0     # noise power (dBm)
SIGMA2       = 10 ** ((NOISE_dBm - 30) / 10)  # Watts

# Gauss-Markov channel parameters
FD_Hz        = 3000.0    # max Doppler frequency (Hz)
TS           = 0.01      # time-slot duration (s)

# ─────────────────────────────────────────────
# STELLAR Hyperparameters
# ─────────────────────────────────────────────
N_POP        = 15    # population size N
N_PARENTS    = 10    # parents per generation Np
N_GEN        = 50    # number of generations Ng
MAX_ATTEMPTS = 5     # max validation attempts Em
R_MIN        = 0.1   # minimum per-user rate (bps/Hz)

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"

# ─────────────────────────────────────────────
# Experiment ranges (to reproduce Figs 3-4)
# ─────────────────────────────────────────────
K_RANGE      = [2, 3, 4, 5, 6]      # Fig 3a, Fig 4
Nt_RANGE     = [8, 10, 12, 14, 16]  # Fig 3b
N_REALIZ     = 3   # channel realisations to average (set higher for smoother curves)
SNR_DB = 30.0   # per-user SNR in dB — tune this to match paper's rate scale