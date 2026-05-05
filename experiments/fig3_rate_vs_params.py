"""
Figure 3 — Achievable Rate vs System Parameters.

3a: Rate vs number of GSs    (K ∈ {2,3,4,5,6}, Nt=12)
3b: Rate vs number of antennas (Nt ∈ {8,10,12,14,16}, K=4)

Compares: STELLAR, Vanilla PPO, Random
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import Pt, SIGMA2, N_GEN, N_POP, N_PARENTS, MAX_ATTEMPTS, R_MIN, N_REALIZ, SNR_DB
from config import K_RANGE, Nt_RANGE
from src.channel   import LEOChannelModel
from src.stellar   import STELLAR
from src.baselines import random_search, vanilla_ppo


def _run_single(H, Pt, K, n_gen, ppo_eps=2000, seed=0):
    """Run all three methods on one channel realisation. Returns (stellar, ppo, rand) rates."""
    sigma2 = SIGMA2

    # STELLAR
    opt = STELLAR(H, sigma2, Pt, K, N=N_POP, Np=N_PARENTS,
                  Ng=n_gen, Em=MAX_ATTEMPTS, R_min=R_MIN)
    _, stellar_rate, _, _ = opt.optimise()

    # PPO (short run for bar comparison)
    _, ppo_rate, _ = vanilla_ppo(H, sigma2, Pt, K, R_min=R_MIN,
                                  n_episodes=ppo_eps,
                                  track_every=ppo_eps)
    # Random
    _, rand_rate, _ = random_search(H, sigma2, Pt, K, R_min=R_MIN,
                                     n_trials=ppo_eps, track_every=ppo_eps)

    return stellar_rate, ppo_rate, rand_rate


def run_fig3a(Nt=12, save_dir="results", n_gen=20, ppo_eps=2000):
    """Rate vs number of GSs (Fig. 3a)."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fig 3a — Rate vs K  |  Nt={Nt}")
    print(f"{'='*60}")

    stellar_rates, ppo_rates, rand_rates = [], [], []

    for K in K_RANGE:
        print(f"\n  K = {K} …")
        s_list, p_list, r_list = [], [], []

        for seed in range(N_REALIZ):
            ch = LEOChannelModel(Nt, K, seed=seed * 100, snr_db=SNR_DB)
            H  = ch.get_H()
            s, p, r = _run_single(H, Pt, K, n_gen, ppo_eps, seed)
            s_list.append(s); p_list.append(p); r_list.append(r)

        stellar_rates.append(np.mean(s_list))
        ppo_rates.append(np.mean(p_list))
        rand_rates.append(np.mean(r_list))
        print(f"  STELLAR={stellar_rates[-1]:.3f}  PPO={ppo_rates[-1]:.3f}  Rand={rand_rates[-1]:.3f}")

    # ── Plot ─────────────────────────────────────────────────────
    x     = np.arange(len(K_RANGE))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width, stellar_rates, width, label="STELLAR",    color="#2196F3")
    bars2 = ax.bar(x,         ppo_rates,     width, label="Vanilla PPO",color="#FF9800")
    bars3 = ax.bar(x + width, rand_rates,    width, label="Random",     color="#9E9E9E")

    # Annotate % gain over PPO
    for i, (s, p) in enumerate(zip(stellar_rates, ppo_rates)):
        if p > 0:
            gain = (s - p) / p * 100
            ax.text(x[i] - width / 2, max(s, p) + 0.3, f"{gain:.1f}%",
                    ha="center", va="bottom", fontsize=8, color="#2196F3")

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_RANGE])
    ax.set_xlabel("Number of Ground Stations (K)", fontsize=12)
    ax.set_ylabel("Achievable Sum Rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Fig. 3a — Rate vs K  (Nt={Nt})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(save_dir, "fig3a_rate_vs_K.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {path}")

    return {"K_range": K_RANGE, "stellar": stellar_rates,
            "ppo": ppo_rates, "random": rand_rates}


def run_fig3b(K=4, save_dir="results", n_gen=20, ppo_eps=2000):
    """Rate vs number of antennas (Fig. 3b)."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fig 3b — Rate vs Nt  |  K={K}")
    print(f"{'='*60}")

    stellar_rates, ppo_rates, rand_rates = [], [], []

    for Nt in Nt_RANGE:
        print(f"\n  Nt = {Nt} …")
        s_list, p_list, r_list = [], [], []

        for seed in range(N_REALIZ):
            ch = LEOChannelModel(Nt, K, seed=seed * 100, snr_db=SNR_DB)
            H  = ch.get_H()
            s, p, r = _run_single(H, Pt, K, n_gen, ppo_eps, seed)
            s_list.append(s); p_list.append(p); r_list.append(r)

        stellar_rates.append(np.mean(s_list))
        ppo_rates.append(np.mean(p_list))
        rand_rates.append(np.mean(r_list))
        print(f"  STELLAR={stellar_rates[-1]:.3f}  PPO={ppo_rates[-1]:.3f}  Rand={rand_rates[-1]:.3f}")

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Nt_RANGE, stellar_rates, "-o",  label="STELLAR",    lw=2)
    ax.plot(Nt_RANGE, ppo_rates,     "--s", label="Vanilla PPO",lw=2)
    ax.plot(Nt_RANGE, rand_rates,    ":^",  label="Random",     lw=2)

    ax.set_xlabel("Number of Transmit Antennas (Nt)", fontsize=12)
    ax.set_ylabel("Achievable Sum Rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Fig. 3b — Rate vs Nt  (K={K})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    path = os.path.join(save_dir, "fig3b_rate_vs_Nt.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {path}")

    return {"Nt_range": Nt_RANGE, "stellar": stellar_rates,
            "ppo": ppo_rates, "random": rand_rates}


if __name__ == "__main__":
    run_fig3a()
    run_fig3b()
