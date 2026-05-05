"""
Figure 2 — Convergence comparison.

Reproduces Fig. 2 of the paper:
    STELLAR (LLM) vs Vanilla PPO vs Random — sum rate vs iterations/episodes.
    System: K=4, Nt=16
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import Pt, SIGMA2, N_GEN, N_POP, N_PARENTS, MAX_ATTEMPTS, R_MIN, SNR_DB
from src.channel   import LEOChannelModel
from src.stellar   import STELLAR
from src.baselines import random_search, vanilla_ppo


def run_fig2(Nt=16, K=4, seed=42, ppo_episodes=30_000, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    ch = LEOChannelModel(Nt, K, seed=seed, snr_db=SNR_DB)
    H  = ch.get_H()

    print("\nRunning STELLAR …")
    optimizer = STELLAR(H, SIGMA2, Pt, K,
                        N=N_POP, Np=N_PARENTS,
                        Ng=N_GEN, Em=MAX_ATTEMPTS, R_min=R_MIN)
    _, _, stellar_history, _ = optimizer.optimise()
    while len(stellar_history) < N_GEN + 1:
        stellar_history.append(stellar_history[-1])

    print("\n Running Vanilla PPO …")
    _, _, ppo_history = vanilla_ppo(
        H, SIGMA2, Pt, K, R_min=R_MIN,
        n_episodes=ppo_episodes,
        track_every=max(1, ppo_episodes // (N_GEN + 1)),
    )
    # Interpolate/subsample to same length as STELLAR history
    ppo_x = np.linspace(0, len(ppo_history) - 1, N_GEN + 1).astype(int)
    ppo_aligned = [ppo_history[i] for i in ppo_x]

    print("\nrunning Random baseline …")
    _, _, rand_history = random_search(
        H, SIGMA2, Pt, K, R_min=R_MIN,
        n_trials=ppo_episodes,
        track_every=max(1, ppo_episodes // (N_GEN + 1)),
    )
    rand_x = np.linspace(0, len(rand_history) - 1, N_GEN + 1).astype(int)
    rand_aligned = [rand_history[i] for i in rand_x]


    iters = np.arange(len(stellar_history))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(iters, stellar_history,  "-o",  label="STELLAR",    lw=2, ms=4)
    ax.plot(iters, ppo_aligned,      "--s", label="Vanilla PPO",lw=2, ms=4)
    ax.plot(iters, rand_aligned,     ":^",  label="Random",     lw=2, ms=4)

    # Replace the annotation block with:
    stellar_final = stellar_history[-1]
    ppo_final = ppo_aligned[-1]
    if ppo_final > 0:
        gap = (stellar_final - ppo_final) / ppo_final * 100
        mid_x = len(iters) - 4
        ax.annotate("",
                    xy=(mid_x, ppo_final),
                    xytext=(mid_x, stellar_final),
                    arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
        ax.text(mid_x + 0.4, (stellar_final + ppo_final) / 2,
                f"{gap:.1f}%\ngain", fontsize=9, color="green", va="center")

    ax.set_xlabel("Iterations / Episodes (×10³ for PPO)", fontsize=12)
    ax.set_ylabel("Achievable Sum Rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Fig. 2 — Convergence Comparison  (Nt={Nt}, K={K})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    path = os.path.join(save_dir, "fig2_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {path}")

    return {
        "stellar_history": stellar_history,
        "ppo_history": ppo_aligned,
        "rand_history": rand_aligned,
    }


if __name__ == "__main__":
    run_fig2()
