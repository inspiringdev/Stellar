"""
Figure 5 — Population Convergence.

Reproduces Fig. 5:  best and worst individuals in the parent set
                    over generations.  (Nt=16, K=4)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from config import Pt, SIGMA2, N_GEN, N_POP, N_PARENTS, MAX_ATTEMPTS, R_MIN, SNR_DB

from src.channel import LEOChannelModel
from src.stellar import STELLAR


def run_fig5(Nt=16, K=4, seed=42, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fig 5 — Population Convergence  |  Nt={Nt}, K={K}")
    print(f"{'='*60}")

    ch = LEOChannelModel(Nt, K, seed=seed, snr_db=SNR_DB)
    H  = ch.get_H()

    opt = STELLAR(H, SIGMA2, Pt, K,
                  N=N_POP, Np=N_PARENTS,
                  Ng=N_GEN, Em=MAX_ATTEMPTS, R_min=R_MIN)
    _, best_rate, best_hist, worst_hist = opt.optimise()

    episodes = np.arange(len(best_hist))

    # Final gap
    final_gap_pct = (best_hist[-1] - worst_hist[-1]) / max(best_hist[-1], 1e-9) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(episodes, best_hist,  "-o", label="Best individual",  lw=2, ms=4)
    ax.plot(episodes, worst_hist, "--s", label="Worst individual", lw=2, ms=4)

    # Mark final gap
    ax.annotate(
        "", xy=(episodes[-1], worst_hist[-1]),
        xytext=(episodes[-1], best_hist[-1]),
        arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
    )
    ax.text(episodes[-1] + 0.3, (best_hist[-1] + worst_hist[-1]) / 2,
            f"{final_gap_pct:.1f}%\nrange", fontsize=9, color="red")

    # Mark where LLM offspring replace random individuals (≈ Np iterations)
    ax.axvline(x=N_PARENTS, color="gray", ls=":", lw=1.2,
               label=f"Initial random replaced (iter≈{N_PARENTS})")

    ax.set_xlabel("Episodes (Generations)", fontsize=12)
    ax.set_ylabel("Achievable Rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Fig. 5 — Population Convergence  (Nt={Nt}, K={K})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    path = os.path.join(save_dir, "fig5_population_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nFinal best rate : {best_hist[-1]:.4f} bps/Hz")
    print(f"Individual range: {final_gap_pct:.1f}%")
    print(f"Saved → {path}")

    return {"best": best_hist, "worst": worst_hist, "final_gap_pct": final_gap_pct}


if __name__ == "__main__":
    run_fig5()
