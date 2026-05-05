"""
STELLAR — LLM-Assisted Evolutionary Optimisation for RSMA in LEO Networks.

Implements Algorithm 1 from the paper.

Phases
------
1. Initialisation : random feasible population of size N
2. Evolution      : LLM generates Ng new offspring via prompt-guided search
3. Update         : keep top-N individuals by fitness
"""

import logging
import numpy as np
from typing import Optional

from src.rsma import compute_rates_rsma, check_constraints
from src.llm_operator import llm_generate_offspring

logger = logging.getLogger(__name__)


class STELLAR:
    """
    Parameters
    ----------
    H       : (K, Nt) channel matrix
    sigma2  : noise power (Watts)
    Pt      : total transmit power budget (Watts)
    K       : number of ground stations
    N       : population size
    Np      : parent-set size
    Ng      : number of generations (= number of LLM calls)
    Em      : max validation attempts per generation
    R_min   : minimum per-user rate (bps/Hz)
    """

    def __init__(
        self,
        H: np.ndarray,
        sigma2: float,
        Pt: float,
        K: int,
        N: int = 15,
        Np: int = 10,
        Ng: int = 30,
        Em: int = 5,
        R_min: float = 0.5,
    ):
        self.H = H
        self.sigma2 = sigma2
        self.Pt = Pt
        self.K = K
        self.N = N
        self.Np = Np
        self.Ng = Ng
        self.Em = Em
        self.R_min = R_min

        self.population: list[np.ndarray] = []
        self.rates: np.ndarray = np.array([])
        self.fitness: np.ndarray = np.array([])

        # Convergence history (one entry per generation)
        self.best_history: list[float] = []
        self.worst_history: list[float] = []   # for Fig. 5

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _random_feasible(self) -> np.ndarray:
        for _ in range(200):
            # Pick one of 4 alpha strategies randomly
            strategy = np.random.randint(0, 4)
            if strategy == 0:
                alpha = np.ones(self.K + 1)
            elif strategy == 1:
                alpha = np.array([0.5] + [2.0] * self.K)
            elif strategy == 2:
                alpha = np.array([2.0] + [0.5] * self.K)
            else:
                alpha = np.random.uniform(0.3, 3.0, self.K + 1)

            fracs = np.random.dirichlet(alpha)
            scale = np.random.uniform(0.5, 1.0)
            p = fracs * self.Pt * scale
            ok, _ = check_constraints(p, self.Pt, self.H, self.sigma2, self.R_min)
            if ok:
                return p
        return np.full(self.K + 1, self.Pt / (self.K + 1))

    def _eval(self, p: np.ndarray) -> float:
        rate, _ = compute_rates_rsma(self.H, p, self.sigma2, self.R_min)
        return rate

    def _compute_fitness(self) -> np.ndarray:
        """
        Normalised fitness per Eq. (17–18).
        f(p_j) = (exp(R_j) − exp(Λ)) / (exp(R_j) + exp(Λ))  ∈ [0, 1)
        """
        R = self.rates
        lam = R.min()
        exp_R = np.exp(np.clip(R - lam, 0, 500))   # numerical stability
        fitness = (exp_R - 1.0) / (exp_R + 1.0)
        return fitness

    def _rescale_to_feasible(self, p: np.ndarray) -> np.ndarray:
        """Ensure p satisfies the power constraint by rescaling if needed."""
        p = np.abs(p)
        total = p.sum()
        if total > self.Pt:
            p = p * self.Pt / total * 0.99
        return p

    # ──────────────────────────────────────────────────────────────
    # Phase 1 — Initialisation
    # ──────────────────────────────────────────────────────────────

    def initialise(self):
        """Populate the initial random population (Algorithm 1, Step 1)."""
        logger.info("Initialising population (N=%d) …", self.N)
        self.population = [self._random_feasible() for _ in range(self.N)]
        self.rates = np.array([self._eval(p) for p in self.population])
        self.fitness = self._compute_fitness()

        best = self.rates.max()
        worst = self.rates.min()
        self.best_history.append(best)
        self.worst_history.append(worst)
        logger.info("Initial best rate: %.4f bps/Hz", best)

    # ──────────────────────────────────────────────────────────────
    # Phases 2 & 3 — Evolution + Update
    # ──────────────────────────────────────────────────────────────

    def run_evolution(self) -> tuple[np.ndarray, float]:
        """
        Iterative evolution loop (Algorithm 1, Steps 2–3).

        Returns
        -------
        p_star : optimal power allocation
        rate   : achieved sum rate (bps/Hz)
        """
        for gen in range(self.Ng):
            # ── Select parent set  (Eq. 19) ──────────────────────
            threshold = np.median(self.fitness)
            eligible = np.where(self.fitness > threshold)[0].tolist()
            if len(eligible) < 2:
                eligible = list(range(len(self.population)))

            n_sel = min(self.Np, len(eligible))
            sel_idx = np.random.choice(eligible, n_sel, replace=False)
            parent_set   = [self.population[i] for i in sel_idx]
            parent_rates = [self.rates[i]      for i in sel_idx]

            # Normalise solutions for LLM prompt
            parent_fracs = [p / self.Pt for p in parent_set]

            # ── Generate offspring  (Eq. 20) ─────────────────────
            offspring: Optional[np.ndarray] = None

            for attempt in range(self.Em):
                raw = llm_generate_offspring(self.K, parent_fracs, parent_rates)

                if raw is None:
                    # LLM unavailable — use guided random perturbation
                    best_p = self.population[int(self.rates.argmax())]
                    raw = best_p / self.Pt + np.random.randn(self.K + 1) * 0.02  # tighter perturbation
                    raw = np.clip(raw, 0.01, None)

                p_cand = self._rescale_to_feasible(raw * self.Pt)
                ok, err = check_constraints(p_cand, self.Pt, self.H, self.sigma2, self.R_min)

                if ok:
                    offspring = p_cand
                    break
                else:
                    logger.debug("Gen %d attempt %d: validation Error %d", gen + 1, attempt + 1, err)

            if offspring is None:
                offspring = self._random_feasible()

            # ── Validate & integrate  (Eq. 22) ───────────────────
            r_off = self._eval(offspring)
            self.population.append(offspring)
            self.rates = np.append(self.rates, r_off)

            # ── Update: keep top-N  (Eq. 23) ─────────────────────
            top_idx = np.argsort(self.rates)[-self.N:]
            self.population = [self.population[i] for i in top_idx]
            self.rates      = self.rates[top_idx]
            self.fitness    = self._compute_fitness()

            best  = self.rates.max()
            worst = self.rates.min()
            self.best_history.append(best)
            self.worst_history.append(worst)
            logger.info("Gen %3d/%d | best=%.4f | worst=%.4f bps/Hz",
                        gen + 1, self.Ng, best, worst)

        # ── Final result  (Eq. 24) ───────────────────────────────
        best_idx = int(self.rates.argmax())
        p_star   = self.population[best_idx]
        return p_star, self.rates[best_idx]

    # ──────────────────────────────────────────────────────────────
    # Convenience: run full pipeline
    # ──────────────────────────────────────────────────────────────

    def optimise(self) -> tuple[np.ndarray, float, list[float], list[float]]:
        """
        Run the full STELLAR pipeline.

        Returns
        -------
        p_star       : optimal power allocation
        best_rate    : achieved sum rate
        best_history : per-generation best rates  (length = Ng + 1)
        worst_history: per-generation worst rates (for Fig. 5)
        """
        self.initialise()
        p_star, best_rate = self.run_evolution()
        return p_star, best_rate, self.best_history, self.worst_history
