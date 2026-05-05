"""
Baseline algorithms:
  1. Random search
  2. Vanilla PPO  (continuous-action, PyTorch implementation)

Both reproduce the baselines from Fig. 2 and Fig. 3 of the paper.
Vanilla PPO requires `torch` (pip install torch --break-system-packages).
"""

import numpy as np
import logging
from src.rsma import compute_rates_rsma, check_constraints

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 1. Random Search
# ══════════════════════════════════════════════════════════════════════

def random_search(
    H: np.ndarray,
    sigma2: float,
    Pt: float,
    K: int,
    R_min: float = 0.5,
    n_trials: int = 500,
    track_every: int = 1,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Random search baseline.

    Returns
    -------
    best_p    : best power allocation found
    best_rate : achieved sum rate
    history   : best-so-far rate tracked every `track_every` trials
    """
    best_rate = 0.0
    best_p = np.full(K + 1, Pt / (K + 1))
    history: list[float] = []

    for t in range(n_trials):
        fracs = np.random.dirichlet(np.ones(K + 1))
        scale = np.random.uniform(0.4, 1.0)
        p = fracs * Pt * scale

        ok, _ = check_constraints(p, Pt, H, sigma2, R_min)
        if ok:
            rate, _ = compute_rates_rsma(H, p, sigma2, R_min)
            if rate > best_rate:
                best_rate = rate
                best_p = p

        if t % track_every == 0:
            history.append(best_rate)

    return best_p, best_rate, history


# ══════════════════════════════════════════════════════════════════════
# 2. Vanilla PPO
# ══════════════════════════════════════════════════════════════════════

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        return torch, nn, optim
    except ImportError:
        return None, None, None


class _ActorCritic:
    """Lightweight Actor-Critic network for continuous power allocation."""

    def __init__(self, state_dim, action_dim, hidden=128):
        torch, nn, _ = _try_import_torch()
        assert torch is not None, "PyTorch required for PPO baseline"
        self.torch = torch
        self.nn = nn

        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
            nn.Linear(hidden, action_dim),  # outputs in (0,1)
        )
        self.actor_log_std = torch.nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def parameters(self):
        import itertools
        return itertools.chain(
            self.actor_mean.parameters(),
            [self.actor_log_std],
            self.critic.parameters(),
        )


def vanilla_ppo(
    H: np.ndarray,
    sigma2: float,
    Pt: float,
    K: int,
    R_min: float = 0.5,
    n_episodes: int = 500_000,
    track_every: int = 500,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    epochs_per_update: int = 4,
    batch_size: int = 64,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Vanilla PPO for continuous power allocation.

    The state is the flattened real/imaginary parts of H.
    The action is a (K+1)-dimensional vector in (0,1) scaled to [0, Pt].

    Note: the paper trains PPO for 5×10^5 episodes. This function
    runs a shorter version for comparison. Set n_episodes=500_000
    for full reproduction (takes ~hours).

    Returns
    -------
    best_p    : best allocation found
    best_rate : achieved sum rate
    history   : tracked best-so-far rates
    """
    torch, nn, optim = _try_import_torch()
    if torch is None:
        logger.warning("PyTorch not available — using random search as PPO proxy.")
        return random_search(H, sigma2, Pt, K, R_min, n_episodes, track_every)

    state_dim  = 2 * H.size   # real + imag parts of H flattened
    action_dim = K + 1

    # Build network
    ac = _ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(ac.parameters(), lr=lr)

    def get_state():
        flat = np.concatenate([H.real.ravel(), H.imag.ravel()])
        flat = (flat - flat.mean()) / (flat.std() + 1e-8)
        return torch.FloatTensor(flat)

    def get_action(state):
        raw = ac.actor_mean(state)
        # Use softmax instead of sigmoid — guarantees sum=1, fixes action mismatch
        action = torch.softmax(raw, dim=-1)
        std = torch.exp(ac.actor_log_std).clamp(1e-4, 0.3)
        dist = torch.distributions.Normal(action, std)
        sample = dist.sample()
        sample = torch.softmax(sample, dim=-1)  # re-normalize sample
        log_prob = dist.log_prob(sample).sum()
        return sample, log_prob

    def reward_fn(action_np):
        p = action_np * Pt
        rate, feasible = compute_rates_rsma(H, p, sigma2, R_min)
        return rate if feasible else -2.0

    best_rate = 0.0
    best_p    = np.full(K + 1, Pt / (K + 1))
    history: list[float] = []

    # Rollout buffer
    states_buf, actions_buf, logp_buf, rew_buf = [], [], [], []

    for ep in range(n_episodes):
        state = get_state()
        action, log_prob = get_action(state)
        action_np = action.detach().numpy()

        r = reward_fn(action_np)
        if r > best_rate:
            best_rate = r
            best_p = action_np * Pt

        states_buf.append(state)
        actions_buf.append(action)
        logp_buf.append(log_prob)
        rew_buf.append(torch.tensor(r, dtype=torch.float32))

        # PPO update every batch_size episodes
        if len(rew_buf) >= batch_size:
            returns = []
            G = 0.0
            for r_ in reversed(rew_buf):
                G = float(r_) + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            states_t  = torch.stack(states_buf)
            actions_t = torch.stack(actions_buf)
            logp_old  = torch.stack(logp_buf).detach()

            for _ in range(epochs_per_update):
                mean_new = ac.actor_mean(states_t)
                std_new  = torch.exp(ac.actor_log_std).clamp(1e-4, 1.0)
                dist_new = torch.distributions.Normal(mean_new, std_new)
                logp_new = dist_new.log_prob(
                    torch.logit(actions_t.clamp(1e-6, 1 - 1e-6))
                ).sum(dim=1)

                ratio = torch.exp(logp_new - logp_old)
                values = ac.critic(states_t).squeeze()
                adv = (returns - values.detach())

                actor_loss = -(
                    torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
                ).mean()
                critic_loss = nn.functional.mse_loss(values, returns)
                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(ac.parameters()), 0.5)
                optimizer.step()

            states_buf.clear(); actions_buf.clear()
            logp_buf.clear();   rew_buf.clear()

        if ep % track_every == 0:
            history.append(best_rate)
            logger.debug("PPO ep %d | best=%.4f", ep, best_rate)

    return best_p, best_rate, history
