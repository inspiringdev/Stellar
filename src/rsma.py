"""
RSMA Rate Computation
Implements Equations (6)–(12) from the paper.

SINR_c_k  = |h_k^H w_c|^2 / (Σ_j |h_k^H w_p_j|^2 + σ^2)       Eq. (6)
SINR_p_k  = |h_k^H w_p_k|^2 / (Σ_{j≠k} |h_k^H w_p_j|^2 + σ^2) Eq. (7)
R_c_k     = log2(1 + SINR_c_k)                                    Eq. (8)
R_p_k     = log2(1 + SINR_p_k)                                    Eq. (9)

Sum rate  = Σ_k c_k + Σ_k R_p_k,   Σ_k c_k ≤ min_k R_c_k       Eq. (10-11)

To maximise sum rate, we use all available common rate:
    Σ_k c_k = min_k R_c_k
    c_k distribution: first satisfy R_min per user, then split remainder.
"""

import numpy as np
from src.beamforming import get_beamforming_vectors


def compute_sinr(H: np.ndarray, w_c: np.ndarray, W_p: np.ndarray, sigma2: float):
    """
    Compute common and private SINRs for all users.

    Returns
    -------
    SINR_c : ndarray (K,) — common stream SINRs
    SINR_p : ndarray (K,) — private stream SINRs
    """
    K = H.shape[0]
    SINR_c = np.zeros(K)
    SINR_p = np.zeros(K)

    for k in range(K):
        hk = H[k]   # (Nt,)

        # Interference from all private streams
        priv_power = sum(abs(hk.conj() @ W_p[j]) ** 2 for j in range(K))

        # Common SINR (Eq. 6)
        num_c = abs(hk.conj() @ w_c) ** 2
        SINR_c[k] = num_c / (priv_power + sigma2)

        # Private SINR (Eq. 7)
        num_p = abs(hk.conj() @ W_p[k]) ** 2
        inter_p = priv_power - abs(hk.conj() @ W_p[k]) ** 2
        SINR_p[k] = num_p / (inter_p + sigma2)

    return SINR_c, SINR_p


def compute_rates_rsma(
    H: np.ndarray,
    p: np.ndarray,
    sigma2: float,
    R_min: float = 0.5,
) -> tuple[float, bool]:
    K = H.shape[0]
    p = np.clip(p, 0, None)

    w_c, W_p = get_beamforming_vectors(H, p)
    SINR_c, SINR_p = compute_sinr(H, w_c, W_p, sigma2)

    Rc = np.log2(1.0 + SINR_c)   # (K,)
    Rp = np.log2(1.0 + SINR_p)   # (K,)

    Rc_total = Rc.min()           # Eq. (10) — bottleneck common rate

    slack = np.maximum(0.0, R_min - Rp)

    if slack.sum() > Rc_total + 1e-9:
        return 0.0, False

    # Distribute remainder equally, clamp, renormalize
    c = slack.copy()
    c += (Rc_total - slack.sum()) / K
    c = np.maximum(c, 0.0)
    if c.sum() > 0:
        c = c * Rc_total / c.sum()

    sum_rate = float(Rc_total + Rp.sum())
    return sum_rate, True


def compute_rates_sdma(
    H: np.ndarray,
    p_priv: np.ndarray,
    sigma2: float,
) -> float:
    """
    Compute SDMA sum rate (no common stream, pure ZF precoding).

    Parameters
    ----------
    H      : (K, Nt)
    p_priv : (K,) private powers
    sigma2 : noise power

    Returns
    -------
    sum_rate : float
    """
    from src.beamforming import compute_beam_directions

    K, Nt = H.shape
    _, W_tilde_p = compute_beam_directions(H)

    W_p = np.array([np.sqrt(float(p_priv[k])) * W_tilde_p[k] for k in range(K)])

    sum_rate = 0.0
    for k in range(K):
        hk = H[k]
        num_p  = abs(hk.conj() @ W_p[k]) ** 2
        inter  = sum(abs(hk.conj() @ W_p[j]) ** 2 for j in range(K) if j != k)
        SINR_p = num_p / (inter + sigma2)
        sum_rate += np.log2(1.0 + SINR_p)

    return float(sum_rate)


# ─────────────────────────────────────────────────────────────────────
# Constraint checker
# ─────────────────────────────────────────────────────────────────────

def check_constraints(
    p: np.ndarray,
    Pt: float,
    H: np.ndarray,
    sigma2: float,
    R_min: float = 0.5,
) -> tuple[bool, int]:
    """
    Validate a power allocation against (12b)–(12d).

    Returns
    -------
    valid     : bool
    error_id  : int  — 0=ok, 1=power, 2=rate, 3=common-rate
    """
    # Constraint (12b): total power
    if np.any(p < -1e-8) or p.sum() > Pt * 1.001:
        return False, 1

    _, feasible = compute_rates_rsma(H, p, sigma2, R_min)
    if not feasible:
        return False, 2

    return True, 0
