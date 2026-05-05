"""
Figure 4 — STELLAR with RSMA vs SDMA.

Reproduces Fig. 4:  Achievable Rate vs GS number,
                    STELLAR+RSMA vs STELLAR+SDMA  (Nt=12, K ∈ {2,3,4,5,6})
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import Pt, SIGMA2, N_GEN, N_POP, N_PARENTS, MAX_ATTEMPTS, R_MIN, N_REALIZ, SNR_DB

from config import K_RANGE
from src.channel   import LEOChannelModel
from src.stellar   import STELLAR
from src.rsma      import compute_rates_sdma
from src.beamforming import compute_beam_directions


def _sdma_stellar(H, Pt, K, n_gen=20, em=5):
    """
    Run STELLAR optimising SDMA power allocation.
    For SDMA: p = [p_1, …, p_K]  (no common stream), objective = SDMA sum rate.
    """
    sigma2 = SIGMA2

    # Monkey-patch a STELLAR instance to use SDMA objective
    import numpy as np
    from src.rsma import check_constraints as _cc

    class STELLAR_SDMA(STELLAR):
        def _eval(self, p):
            # p here is (K,) private powers
            return compute_rates_sdma(self.H, p, self.sigma2)

        def _random_feasible(self):
            for _ in range(200):
                fracs = np.random.dirichlet(np.ones(K))
                scale = np.random.uniform(0.4, 1.0)
                p = fracs * Pt * scale
                if p.sum() <= Pt and np.all(p >= 0):
                    return p
            return np.full(K, Pt / K)

        def _rescale_to_feasible(self, p):
            p = np.abs(p)
            total = p.sum()
            if total > self.Pt:
                p = p * self.Pt / total * 0.99
            return p

    from src.llm_operator import llm_generate_offspring as _llm

    class SDMAOptimizer(STELLAR_SDMA):
        def run_evolution(self):
            for gen in range(self.Ng):
                threshold = np.median(self.fitness) if len(self.fitness) else 0.0
                eligible = np.where(self.fitness > threshold)[0].tolist() if len(self.fitness) else []
                if len(eligible) < 2:
                    eligible = list(range(len(self.population)))

                n_sel = min(self.Np, len(eligible))
                sel_idx = np.random.choice(eligible, n_sel, replace=False)
                parent_set   = [self.population[i] for i in sel_idx]
                parent_rates = [self.rates[i] for i in sel_idx]
                parent_fracs = [p / self.Pt for p in parent_set]

                offspring = None
                for _ in range(self.Em):
                    raw = _llm(K, parent_fracs, parent_rates)   # K variables
                    if raw is None:
                        best_p = self.population[int(self.rates.argmax())]
                        raw = best_p / self.Pt + np.random.randn(K) * 0.05
                        raw = np.clip(raw, 0, None)
                    raw = np.array(raw).ravel()
                    if len(raw) > K:
                        raw = raw[:K]
                    elif len(raw) < K:
                        raw = np.pad(raw, (0, K - len(raw)), constant_values=0.05)

                    p_cand = self._rescale_to_feasible(raw * self.Pt)
                    if p_cand.sum() <= self.Pt and np.all(p_cand >= 0):
                        offspring = p_cand
                        break

                if offspring is None:
                    offspring = self._random_feasible()

                r_off = self._eval(offspring)
                self.population.append(offspring)
                self.rates = np.append(self.rates, r_off)

                top_idx = np.argsort(self.rates)[-self.N:]
                self.population = [self.population[i] for i in top_idx]
                self.rates = self.rates[top_idx]
                self.fitness = self._compute_fitness()

                self.best_history.append(self.rates.max())
                self.worst_history.append(self.rates.min())

            best_idx = int(self.rates.argmax())
            return self.population[best_idx], self.rates[best_idx]

    opt = SDMAOptimizer(H, sigma2, Pt, K, N=N_POP, Np=N_PARENTS,
                        Ng=n_gen, Em=em, R_min=R_MIN)
    # Adjust population initialisation shape for SDMA
    opt.population = [opt._random_feasible() for _ in range(N_POP)]
    opt.rates = np.array([opt._eval(p) for p in opt.population])
    opt.fitness = opt._compute_fitness()

    _, rate = opt.run_evolution()
    return rate


def run_fig4(Nt=12, save_dir="results", n_gen=20):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fig 4 — RSMA vs SDMA  |  Nt={Nt}")
    print(f"{'='*60}")

    rsma_rates = []
    sdma_rates = []

    for K in K_RANGE:
        print(f"\n  K = {K} …")
        r_list, s_list = [], []

        for seed in range(N_REALIZ):
            ch = LEOChannelModel(Nt, K, seed=seed * 100, snr_db=SNR_DB)
            H  = ch.get_H()

            # STELLAR + RSMA
            opt = STELLAR(H, SIGMA2, Pt, K, N=N_POP, Np=N_PARENTS,
                          Ng=n_gen, Em=MAX_ATTEMPTS, R_min=R_MIN)
            _, r_rsma, _, _ = opt.optimise()
            r_list.append(r_rsma)

            # STELLAR + SDMA
            r_sdma = _sdma_stellar(H, Pt, K, n_gen)
            s_list.append(r_sdma)

        rsma_rates.append(np.mean(r_list))
        sdma_rates.append(np.mean(s_list))
        gain = (rsma_rates[-1] - sdma_rates[-1]) / max(sdma_rates[-1], 1e-9) * 100
        print(f"  RSMA={rsma_rates[-1]:.3f}  SDMA={sdma_rates[-1]:.3f}  gain={gain:.1f}%")

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(K_RANGE, rsma_rates, "-o", label="STELLAR with RSMA", lw=2)
    ax.plot(K_RANGE, sdma_rates, "--s", label="STELLAR with SDMA", lw=2)

    for i, (r, s) in enumerate(zip(rsma_rates, sdma_rates)):
        if s > 0:
            gain = (r - s) / s * 100
            ax.annotate(f"{gain:.1f}%",
                        xy=(K_RANGE[i], (r + s) / 2),
                        ha="center", fontsize=8, color="green")

    ax.set_xlabel("Number of Ground Stations (K)", fontsize=12)
    ax.set_ylabel("Achievable Sum Rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Fig. 4 — RSMA vs SDMA  (Nt={Nt})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    path = os.path.join(save_dir, "fig4_rsma_vs_sdma.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {path}")

    return {"K_range": K_RANGE, "rsma": rsma_rates, "sdma": sdma_rates}


if __name__ == "__main__":
    run_fig4()
