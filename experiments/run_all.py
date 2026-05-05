"""
Run all experiments in sequence to reproduce Figs 2–5.

Usage:
    python experiments/run_all.py [--quick]

--quick : uses fewer iterations/seeds (fast demo, less accurate)
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import datetime

from experiments.fig2_convergence      import run_fig2
from experiments.fig3_rate_vs_params   import run_fig3a, run_fig3b
from experiments.fig4_rsma_vs_sdma     import run_fig4
from experiments.fig5_population_convergence import run_fig5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run with reduced parameters for fast demo")
    args = parser.parse_args()

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    if args.quick:
        n_gen = 10
        ppo_eps = 1_000  # ← was 100
    else:
        n_gen = 50
        ppo_eps = 30_000

    all_results = {}
    start = datetime.now()

    print("\n" + "█"*60)
    print("  STELLAR — Paper Reproduction Experiments")
    print("█"*60)

    try:
        print("\n▶ Figure 2 — Convergence comparison")
        all_results["fig2"] = run_fig2(Nt=16, K=4, ppo_episodes=ppo_eps,
                                        save_dir=save_dir)
    except Exception as e:
        print(f"  !! Fig 2 failed: {e}")

    try:
        print("\n▶ Figure 3a — Rate vs K")
        all_results["fig3a"] = run_fig3a(Nt=12, save_dir=save_dir,
                                          n_gen=n_gen, ppo_eps=ppo_eps)
    except Exception as e:
        print(f"  !! Fig 3a failed: {e}")

    try:
        print("\n▶ Figure 3b — Rate vs Nt")
        all_results["fig3b"] = run_fig3b(K=4, save_dir=save_dir,
                                          n_gen=n_gen, ppo_eps=ppo_eps)
    except Exception as e:
        print(f"  !! Fig 3b failed: {e}")

    try:
        print("\n▶ Figure 4 — RSMA vs SDMA")
        all_results["fig4"] = run_fig4(Nt=12, save_dir=save_dir, n_gen=n_gen)
    except Exception as e:
        print(f"  !! Fig 4 failed: {e}")

    try:
        print("\n▶ Figure 5 — Population convergence")
        all_results["fig5"] = run_fig5(Nt=16, K=4, save_dir=save_dir)
    except Exception as e:
        print(f"  !! Fig 5 failed: {e}")

    elapsed = (datetime.now() - start).total_seconds()

    # Save numeric results
    def _make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(v) for v in obj]
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    json_path = os.path.join(save_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(_make_serialisable(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"All done in {elapsed:.1f}s")
    print(f"Figures saved to ./{save_dir}/")
    print(f"Numeric results → {json_path}")
    print("="*60)


if __name__ == "__main__":
    main()
