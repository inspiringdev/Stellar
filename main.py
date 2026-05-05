import os
os.environ["OPENAI_API_KEY"] = "yourkey"  # ← paste your key
os.environ["GEMINI_API_KEY"] = "yourkey"
import sys
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from config import Pt, SIGMA2, N_GEN, N_POP, N_PARENTS, MAX_ATTEMPTS, R_MIN
from src.channel import LEOChannelModel
from src.stellar import STELLAR


def main():
    print("=" * 62)
    print("  STELLAR: LLM-Assisted RSMA Optimisation for LEO Networks")
    print("=" * 62)

    Nt, K = 16, 4
    seed  = 0

    # ── Channel ──────────────────────────────────────────────────
    from config import SNR_DB
    ch = LEOChannelModel(Nt, K, seed=seed, snr_db=SNR_DB)
    H  = ch.get_H()

    print(f"\nSystem  : Nt={Nt}, K={K}, Pt={10*np.log10(Pt*1e3):.0f} dBm")
    print(f"Channel : altitude=780 km, fc=4 GHz, σ²={10*np.log10(SIGMA2*1e3):.0f} dBm")
    print(f"STELLAR : N={N_POP}, Np={N_PARENTS}, Ng={N_GEN}, Em={MAX_ATTEMPTS}")

    api_set = bool(os.environ.get("OPENAI_API_KEY"))
    print(f"\nLLM operator : {'OPENAI API (active)' if api_set else 'guided random (set open ai api to enable LLM)'}")

    # ── Run STELLAR ───────────────────────────────────────────────
    print(f"\nStarting STELLAR optimisation …\n")
    opt = STELLAR(H, SIGMA2, Pt, K, N=N_POP, Np=N_PARENTS,
                  Ng=N_GEN, Em=MAX_ATTEMPTS, R_min=R_MIN)
    p_star, best_rate, best_hist, worst_hist = opt.optimise()

    print(f"\n{'─'*40}")
    print(f"Optimal power allocation:")
    print(f"  p_c  = {p_star[0]*1e3:.2f} mW")
    for k in range(K):
        print(f"  p_{k+1}  = {p_star[k+1]*1e3:.2f} mW")
    print(f"  Total = {p_star.sum()*1e3:.2f} mW  (budget: {Pt*1e3:.0f} mW)")
    print(f"\nAchieved sum rate : {best_rate:.4f} bps/Hz")
    print(f"Initial best rate : {best_hist[0]:.4f} bps/Hz")
    print(f"Convergence gain  : {(best_rate - best_hist[0]) / max(best_hist[0], 1e-9) * 100:.1f}%")
    print("─" * 40)

    # ── Optional: quick plot ──────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(best_hist,  label="Best individual",  lw=2)
        ax.plot(worst_hist, label="Worst individual", lw=2, ls="--")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Sum Rate (bps/Hz)")
        ax.set_title(f"STELLAR Convergence  (Nt={Nt}, K={K})")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        fig.savefig("results/stellar_demo.png", dpi=150)
        plt.close(fig)
        print("\nConvergence plot → results/stellar_demo.png")
    except Exception:
        pass

    print("\nTo reproduce all paper figures, run:")
    print("  python experiments/run_all.py --quick   # fast demo")
    print("  python experiments/run_all.py           # full reproduction")


if __name__ == "__main__":
    main()
