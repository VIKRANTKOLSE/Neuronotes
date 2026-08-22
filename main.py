"""
main.py — Neuronotes SMD-CC-MIRT-KL-CAT Master Experiment Pipeline
==================================================================
Runs the complete adaptive assessment and misconception diagnostics pipeline:

  Step 1: Data processing  (Ingest master CSV → clean artefacts)
  Step 2: Simulation       (Isolated models, reproducible learner pools, multi-seed support)
  Step 3: Save CSVs        (simulation_results.csv, multi_seed_results.csv, rmse_trajectory.csv,
                            calibrated_item_bank.csv)
  Step 4: Plot results     (metric_comparison.png, rmse_comparison.png)

Usage:
  python main.py                                          # Fast default (P7, seed 42)
  python main.py --suite paper                            # Paper comparison set (7 models)
  python main.py --suite full                             # Complete 24-model benchmark
  python main.py --models B4_MIRT P7_Full_Neuronotes     # Custom cherry-picked comparison
  python main.py --suite paper --seeds 42 43 44 45 46     # Multi-seed statistical evaluation
  python main.py --suite paper --learners 1000 --seed 42  # Large-scale single seed
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

# Ensure package is importable when run from codes/
_CODES_DIR = Path(__file__).parent
if str(_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(_CODES_DIR))

import data_processing
from simulation.run_experiment import run_all_stages, save_results, RESULTS_DIR, DATA_DIR
from simulation.metrics import MetricBundle, cohens_d

SYSTEM_SUITES = {
    # Default: the deployable model only, for quick iteration.
    "final": ["P7_Full_Neuronotes"],
    # Paper-facing primary comparison and one-factor ablations.
    "paper": ["B4_MIRT", "P7_Full_Neuronotes", "C1_NoPrereq", "C2_NoCMatrix",
              "C3_FixedC", "C4_NoSMD", "C5_NoRouter"],
    # All baselines, progressive stages, ablations, and policy experiments.
    "full": None,
}

# ============================================================
# Plot helpers
# ============================================================

STAGE_COLORS = {
    "B": "#e74c3c",   # Baselines — red
    "P": "#3498db",   # Progressive — blue
    "C": "#e67e22",   # Ablation — orange
    "D": "#2ecc71",   # Selection — green
    "E": "#9b59b6",   # Feedback — purple
}

def _color_for(name: str) -> str:
    prefix = name[0].upper()
    return STAGE_COLORS.get(prefix, "#95a5a6")


def plot_metric_comparison(df: pd.DataFrame, out_path: Path):
    """Bar chart comparing RMSE, F1, AUROC, efficiency across all systems.

    Displays every system present in `df` — no hardcoded subset filter.
    Figure width scales automatically with the number of models.
    """
    fig_w = max(16, len(df) * 1.3)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 14))
    fig.patch.set_facecolor("#1a1a2e")
    axes = axes.flatten()

    df_plot = df.copy()
    # Normalise efficiency for display (invert so higher=better)
    df_plot["efficiency_q_inv"] = 1.0 / df_plot["efficiency_q"].replace(0, 1)
    plot_cols = ["rmse", "f1", "auroc", "efficiency_q_inv"]
    ytitles   = ["RMSE ↓ (lower = better)", "F1 ↑ (higher = better)", "AUROC ↑ (higher = better)",
                  "1/EfficiencyQ ↑ (higher = fewer questions)"]

    for ax, col, title in zip(axes, plot_cols, ytitles):
        ax.set_facecolor("#16213e")
        colors = [_color_for(n) for n in df_plot["system"]]
        bars   = ax.bar(range(len(df_plot)), df_plot[col], color=colors,
                        edgecolor="white", linewidth=0.5, alpha=0.9)

        # Error bars for RMSE if confidence interval or multi-seed std available
        if col == "rmse" and "rmse_ci95_lo" in df_plot.columns and "rmse_ci95_hi" in df_plot.columns:
            ax.errorbar(range(len(df_plot)),
                        df_plot["rmse"],
                        yerr=[(df_plot["rmse"] - df_plot["rmse_ci95_lo"]).clip(lower=0).values,
                              (df_plot["rmse_ci95_hi"] - df_plot["rmse"]).clip(lower=0).values],
                        fmt="none", color="white", capsize=3.5, linewidth=1.2)

        # Highlight P7 (Full Neuronotes)
        for i, name in enumerate(df_plot["system"]):
            if "P7" in name or "Full" in name:
                bars[i].set_edgecolor("#f1c40f")
                bars[i].set_linewidth(2.8)

        ax.set_xticks(range(len(df_plot)))
        ax.set_xticklabels(df_plot["system"], rotation=65, ha="right",
                           fontsize=8, color="white")
        ax.set_ylabel(ytitles[plot_cols.index(col)], color="white", fontsize=9)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#334155")

    # Legend
    legend_patches = [mpatches.Patch(color=v, label=k + " stages")
                      for k, v in STAGE_COLORS.items()]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=5, facecolor="#1a1a2e", edgecolor="white",
               labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(f"Neuronotes Comparative Assessment ({len(df)} systems)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"Saved {out_path.name}")


def plot_rmse_comparison(df_traj: pd.DataFrame, out_path: Path):
    """RMSE-vs-question-count convergence curves."""
    systems  = [c for c in df_traj.columns if c != "step"]
    fig_w    = max(14, len(systems) * 0.8)
    fig, ax  = plt.subplots(figsize=(fig_w, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(systems), 1))
    for i, sys_name in enumerate(systems):
        if sys_name not in df_traj.columns:
            continue
        y     = df_traj[sys_name].values
        steps = df_traj["step"].values
        lw    = 2.8 if "P7" in sys_name or "Full" in sys_name else 1.4
        ls    = "-" if sys_name.startswith("P") else ("--" if sys_name.startswith("B") else ":")
        ax.plot(steps, y, label=sys_name, linewidth=lw, linestyle=ls,
                color=cmap(i), alpha=0.9)

    ax.axhline(y=0.35, color="#f1c40f", linestyle="--", linewidth=1.2,
               label="RMSE threshold (0.35)", alpha=0.8)
    ax.set_xlabel("Number of Questions", color="white", fontsize=11)
    ax.set_ylabel("Mean RMSE", color="white", fontsize=11)
    ax.set_title("RMSE Convergence: Neuronotes vs Baselines",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#334155")
    ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#334155",
              labelcolor="white", fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"Saved {out_path.name}")


def save_calibrated_item_bank():
    """Copy items_clean.csv → results/calibrated_item_bank.csv."""
    src = DATA_DIR / "items_clean.csv"
    dst = RESULTS_DIR / "calibrated_item_bank.csv"
    if src.exists():
        pd.read_csv(src).to_csv(dst, index=False)
        print(f"Saved calibrated_item_bank.csv ({len(pd.read_csv(dst))} items)")


# ============================================================
# Main Execution
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Neuronotes Adaptive Testing & Psychometric Simulation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          # Fast: P7 only (seed 42)
  python main.py --suite paper                            # Paper comparison set (7 models)
  python main.py --suite full                             # All 24 models
  python main.py --models B4_MIRT P7_Full_Neuronotes     # Cherry-pick models
  python main.py --suite paper --seeds 42 43 44 45 46     # Multi-seed evaluation across 5 seeds
  python main.py --suite paper --learners 500 --seed 1    # Scaled learner population
        """,
    )
    parser.add_argument("--learners", type=int, default=200,
                        help="Number of synthetic learners per model run (default: 200)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip raw data processing and reuse data/ artefacts")
    parser.add_argument("--suite", choices=SYSTEM_SUITES, default="final",
                        help="Experiment suite: final (fast), paper, or full (default: final)")
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Run specific models by name; overrides --suite.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for single run (default: 42)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="List of random seeds to evaluate for multi-seed aggregation")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Run N consecutive seeds starting from --seed (e.g. --n-seeds 5)")
    args = parser.parse_args()

    # Determine seed list
    if args.seeds:
        seeds_to_run = args.seeds
    elif args.n_seeds is not None:
        seeds_to_run = list(range(args.seed, args.seed + max(1, args.n_seeds)))
    else:
        seeds_to_run = [args.seed]

    # --models overrides --suite when explicitly provided
    if args.models:
        systems_to_run = args.models
        print(f"\n[Config] Running {len(systems_to_run)} selected model(s): {', '.join(systems_to_run)}")
    else:
        systems_to_run = SYSTEM_SUITES[args.suite]
        label = "all" if systems_to_run is None else str(len(systems_to_run))
        print(f"\n[Config] Suite='{args.suite}' ({label} models)")

    print(f"[Config] Random seed(s): {seeds_to_run} ({len(seeds_to_run)} run{'s' if len(seeds_to_run)>1 else ''})")
    print(f"[Config] Learners per run: {args.learners}")

    # ---- Step 1: Data processing ----
    if not args.skip_data:
        print("\n[Step 1] Data Processing")
        data_processing.main()
    else:
        print("\n[Step 1] Skipping data processing (--skip-data flag)")

    # ---- Step 2 & 3: Simulation across seeds ----
    print("\n[Step 2] Running Experiments")

    all_seed_bundles = []
    seed_records = []

    for seed_idx, s in enumerate(seeds_to_run, 1):
        if len(seeds_to_run) > 1:
            print(f"\n>>> Running Seed {seed_idx}/{len(seeds_to_run)} (seed={s}) <<<")
        bundles = run_all_stages(n_learners=args.learners, seed=s,
                                 systems_to_run=systems_to_run)
        all_seed_bundles.append(bundles)

        for b in bundles:
            seed_records.append({
                "seed": s,
                "system": b.system_name,
                "n_learners": b.n_learners,
                "mean_questions": round(b.mean_questions, 2),
                "rmse": round(b.rmse, 4),
                "mae": round(b.mae, 4),
                "ece": round(b.ece, 4),
                "precision": round(b.precision, 4),
                "recall": round(b.recall, 4),
                "f1": round(b.f1, 4),
                "auroc": round(b.auroc, 4),
                "auroc_micro": round(b.auroc_micro, 4),
                "valid_misconception_dims": b.valid_misconception_dims,
                "efficiency_q": round(b.efficiency_q, 2),
            })

    df_seeds = pd.DataFrame(seed_records)

    # Save detailed multi-seed breakdown if multiple seeds were run
    if len(seeds_to_run) > 1:
        df_seeds.to_csv(RESULTS_DIR / "multi_seed_results.csv", index=False)
        print(f"\nSaved multi_seed_results.csv ({len(df_seeds)} total runs)")

    # Aggregate across seeds
    summary_rows = []
    grouped = df_seeds.groupby("system", sort=False)
    for sys_name, group in grouped:
        n_seeds = len(group)
        rmse_mean = group["rmse"].mean()
        rmse_std  = group["rmse"].std(ddof=1) if n_seeds > 1 else group.iloc[0].get("rmse_std", 0.0)
        
        # 95% CI calculation
        if n_seeds > 1:
            sem = stats.sem(group["rmse"])
            h = sem * stats.t.ppf(0.975, df=n_seeds - 1)
            ci_lo, ci_hi = rmse_mean - h, rmse_mean + h
        else:
            ci_lo = group["rmse"].iloc[0] - 1.96 * (rmse_std / max(np.sqrt(args.learners), 1))
            ci_hi = group["rmse"].iloc[0] + 1.96 * (rmse_std / max(np.sqrt(args.learners), 1))

        summary_rows.append({
            "system": sys_name,
            "n_seeds": n_seeds,
            "n_learners": int(group["n_learners"].mean()),
            "mean_questions": round(group["mean_questions"].mean(), 2),
            "rmse": round(rmse_mean, 4),
            "rmse_std": round(rmse_std, 4),
            "rmse_ci95_lo": round(ci_lo, 4),
            "rmse_ci95_hi": round(ci_hi, 4),
            "mae": round(group["mae"].mean(), 4),
            "mae_std": round(group["mae"].std(ddof=1) if n_seeds > 1 else 0.0, 4),
            "ece": round(group["ece"].mean(), 4),
            "precision": round(group["precision"].mean(), 4),
            "recall": round(group["recall"].mean(), 4),
            "f1": round(group["f1"].mean(), 4),
            "f1_std": round(group["f1"].std(ddof=1) if n_seeds > 1 else 0.0, 4),
            "auroc": round(group["auroc"].mean(), 4),
            "auroc_std": round(group["auroc"].std(ddof=1) if n_seeds > 1 else 0.0, 4),
            "auroc_micro": round(group["auroc_micro"].mean(), 4),
            "valid_misconception_dims": int(group["valid_misconception_dims"].mean()),
            "efficiency_q": round(group["efficiency_q"].mean(), 2),
        })

    df_results = pd.DataFrame(summary_rows)
    df_results.to_csv(RESULTS_DIR / "simulation_results.csv", index=False)
    print(f"Saved simulation_results.csv ({len(df_results)} systems)")

    # Save trajectory for the first seed (or average)
    last_bundles = all_seed_bundles[0]
    max_len = max(len(b.rmse_trajectory) for b in last_bundles)
    traj_dict = {"step": list(range(1, max_len + 1))}
    for b in last_bundles:
        padded = b.rmse_trajectory + [np.nan] * (max_len - len(b.rmse_trajectory))
        traj_dict[b.system_name] = padded
    pd.DataFrame(traj_dict).to_csv(RESULTS_DIR / "rmse_trajectory.csv", index=False)
    print(f"Saved rmse_trajectory.csv")

    save_calibrated_item_bank()

    # ---- Step 4: Plots ----
    print("\n[Step 4] Generating Plots")
    plot_metric_comparison(df_results, RESULTS_DIR / "metric_comparison.png")
    df_traj = pd.read_csv(RESULTS_DIR / "rmse_trajectory.csv")
    plot_rmse_comparison(df_traj, RESULTS_DIR / "rmse_comparison.png")

    # ---- Results Summary ----
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Neuronotes Psychometric & Diagnostic Benchmark)")
    print("=" * 80)
    display_cols = ["system", "rmse", "mae", "ece", "f1", "auroc", "efficiency_q"]
    if len(seeds_to_run) > 1:
        display_cols = ["system", "rmse", "rmse_std", "f1", "f1_std", "auroc", "auroc_std", "efficiency_q"]
    print(df_results[display_cols].to_string(index=False))

    # Comparative Effect Sizes: P7 vs B4_MIRT
    p7_row = df_results[df_results["system"].str.contains("P7")]
    b4_row = df_results[df_results["system"] == "B4_MIRT"]
    if not p7_row.empty and not b4_row.empty:
        rmse_p7 = float(p7_row["rmse"].values[0])
        rmse_b4 = float(b4_row["rmse"].values[0])
        auroc_p7 = float(p7_row["auroc"].values[0])
        auroc_b4 = float(b4_row["auroc"].values[0])
        f1_p7 = float(p7_row["f1"].values[0])
        f1_b4 = float(b4_row["f1"].values[0])

        delta_rmse = rmse_p7 - rmse_b4
        delta_auroc = auroc_p7 - auroc_b4

        print("\n" + "-" * 60)
        print("PRIMARY HYPOTHESIS COMPARISON: P7 (Neuronotes) vs B4 (Standard MIRT)")
        print("-" * 60)
        print(f"  Delta RMSE  (P7 - B4): {delta_rmse:+.4f}  (lower is better)")
        print(f"  Delta AUROC (P7 - B4): {delta_auroc:+.4f}  (higher is better; B4 has no misconception tracking = 0.500)")
        print(f"  Delta F1    (P7 - B4): {f1_p7 - f1_b4:+.4f}")
        print("-" * 60)

    print(f"\nAll outputs saved in: {RESULTS_DIR}")
    print("  - simulation_results.csv")
    if len(seeds_to_run) > 1:
        print("  - multi_seed_results.csv")
    print("  - rmse_trajectory.csv")
    print("  - calibrated_item_bank.csv")
    print("  - metric_comparison.png")
    print("  - rmse_comparison.png")


if __name__ == "__main__":
    main()
