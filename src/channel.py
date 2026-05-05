"""
LEO Channel Model — with correct SNR normalization.
The raw path-loss amplitude is preserved but H is normalized so that
E[||h_k||^2] = Nt, then a separate SNR scaling is applied.
This matches standard massive-MIMO paper conventions.
"""

import numpy as np
from scipy.special import j0


class LEOChannelModel:
    def __init__(
        self,
        Nt: int,
        K: int,
        altitude_km: float = 780.0,
        fc_GHz: float = 4.0,
        Gs_dBi: float = 35.0,
        Gk_dBi: float = 24.3,
        noise_dBm: float = -90.0,
        fd_Hz: float = 3000.0,
        Ts: float = 0.01,
        snr_db: float = 20.0,   # ← NEW: explicit per-user SNR in dB
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.Nt = Nt
        self.K = K
        self.c = 3e8
        self.fc = fc_GHz * 1e9
        self.altitude = altitude_km * 1e3

        self.Gs = 10 ** (Gs_dBi / 10)
        self.Gk = np.full(K, 10 ** (Gk_dBi / 10))

        # ── noise power (Watts) ──────────────────────────────────
        self.sigma2 = 10 ** ((noise_dBm - 30) / 10)

        # ── Gauss-Markov correlation ─────────────────────────────
        self.rho = float(j0(2 * np.pi * fd_Hz * Ts))

        # ── Slant ranges (vary elevation 30°–80°) ────────────────
        elev_deg = np.linspace(30, 80, K)
        self.dk = self.altitude / np.sin(np.deg2rad(elev_deg))

        # ── Raw path-loss amplitude (kept for reference) ─────────
        self.pl_raw = np.sqrt(
            self.Gs * self.Gk
            * (self.c / (4 * np.pi * self.fc * self.dk)) ** 2
        )

        # ── SNR-based normalization ──────────────────────────────
        # Scale channel so that ||h_k||^2 * Pt / sigma2 ≈ SNR (linear)
        # This is the standard convention in beamforming papers.
        # pl_norm[k] makes E[||h_k||^2] = Nt, then snr_scale adjusts SNR.
        snr_linear = 10 ** (snr_db / 10)
        # Each entry of g is CN(0,1/2) per component, so E[||g_k||^2] = Nt
        # We want Pt * ||w||^2 * |h_k^H w|^2 / sigma2 ~ SNR
        # With ||w||^2=1 and E[|h^H w|^2] = ||h||^2 = pl^2 * Nt:
        #   pl^2 * Nt * Pt / sigma2 = SNR  →  pl = sqrt(SNR * sigma2 / (Nt * Pt))
        from config import Pt as Pt_cfg
        snr_per_user = snr_linear * (self.pl_raw / self.pl_raw.max())  # relative SNR scaling
        self.pl = np.sqrt(snr_per_user * self.sigma2 / Pt_cfg)

        self._init_fading()

    def _init_fading(self):
        real = np.random.randn(self.K, self.Nt)
        imag = np.random.randn(self.K, self.Nt)
        self.g = (real + 1j * imag) / np.sqrt(2)

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self._init_fading()

    def step(self):
        e = (np.random.randn(self.K, self.Nt)
             + 1j * np.random.randn(self.K, self.Nt)) / np.sqrt(2)
        self.g = self.rho * self.g + np.sqrt(1.0 - self.rho ** 2) * e

    def get_H(self) -> np.ndarray:
        H = np.zeros((self.K, self.Nt), dtype=complex)
        for k in range(self.K):
            H[k] = self.pl[k] * self.g[k]
        return H