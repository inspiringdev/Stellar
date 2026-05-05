"""
Beamforming Initialisation — ZF + MRT
Implements Equations (13)–(14) from the paper.

Common stream  : MRT direction  w̃_c  = Σ h_k / |Σ h_k|
Private stream : ZF  direction  w̃_p_k = v_k / |v_k|
                 where V = G^H (G G^H)^{-1}

Full beamforming vectors:
    w_c   = sqrt(pc) * w̃_c
    w_p_k = sqrt(pk) * w̃_p_k
"""

import numpy as np


def compute_beam_directions(H: np.ndarray):
    """
    Compute normalised beam directions for common (MRT) and
    private (ZF) streams.

    Parameters
    ----------
    H : ndarray, shape (K, Nt)
        Channel matrix.

    Returns
    -------
    w_tilde_c : ndarray, shape (Nt,)
        Normalised MRT direction for the common stream.
    W_tilde_p : ndarray, shape (K, Nt)
        Normalised ZF directions for private streams.
    """
    K, Nt = H.shape

    # ── Common stream: MRT ─────────────────────────────────────────
    sum_h = H.sum(axis=0)                        # (Nt,)
    norm = np.linalg.norm(sum_h)
    if norm < 1e-20:
        sum_h = np.random.randn(Nt) + 1j * np.random.randn(Nt)
        norm = np.linalg.norm(sum_h)
    w_tilde_c = sum_h / norm                     # (Nt,)

    # ── Private streams: ZF (pseudo-inverse) ───────────────────────
    # G = H  (K×Nt),  V = G^H (G G^H)^{-1}  shape (Nt, K)
    GGH = H @ H.conj().T                         # (K, K)
    # Add small regularisation for numerical stability
    reg_val = max(1e-6 * np.linalg.norm(GGH), 1e-10)
    reg = reg_val * np.eye(K)
    V = H.conj().T @ np.linalg.inv(GGH + reg)   # (Nt, K)

    W_tilde_p = np.zeros((K, Nt), dtype=complex)
    for k in range(K):
        v_k = V[:, k]
        nv = np.linalg.norm(v_k)
        if nv < 1e-20:
            v_k = np.random.randn(Nt) + 1j * np.random.randn(Nt)
            nv = np.linalg.norm(v_k)
        W_tilde_p[k] = v_k / nv

    return w_tilde_c, W_tilde_p


def get_beamforming_vectors(H: np.ndarray, p: np.ndarray):
    """
    Assemble full precoding vectors from power allocation.

    Parameters
    ----------
    H : ndarray, shape (K, Nt)
    p : ndarray, shape (K+1,)
        p[0] = pc, p[1:] = [p_1, …, p_K]

    Returns
    -------
    w_c  : ndarray, shape (Nt,)   — common stream beamformer
    W_p  : ndarray, shape (K, Nt) — private stream beamformers
    """
    K = H.shape[0]
    pc = float(p[0])
    pk = p[1:]                                    # (K,)

    w_tilde_c, W_tilde_p = compute_beam_directions(H)

    w_c = np.sqrt(pc) * w_tilde_c                # (Nt,)
    W_p = np.array([np.sqrt(float(pk[k])) * W_tilde_p[k] for k in range(K)])  # (K, Nt)

    return w_c, W_p
