"""
main.py — Neuronotes SMD-CC-MIRT-KL-CAT
=========================================
Entry point. Runs the complete pipeline:

  Step 1: Data processing  (neuronotes_final_master_READY.csv → 3 artefacts)
  Step 2: Simulation       (all 5 stages, A–E)
  Step 3: Save CSVs        (simulation_results.csv, rmse_trajectory.csv,
                            calibrated_item_bank.csv)
  Step 4: Plot results     (metric_comparison.png, rmse_comparison.png)

Usage:
  py main.py
  py main.py --learners 50    # quick run for testing
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Ensure package is importable when run from codes/
_CODES_DIR = Path(__file__).parent
if str(_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(_CODES_DIR))

import data_processing
from simulation.run_experiment import run_all_stages, save_results, RESULTS_DIR, DATA_DIR

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
    """Bar chart comparing RMSE, F1, AUROC, efficiency across all systems."""
    metrics   = ["rmse", "f1", "auroc", "efficiency_q"]
    titles    = ["RMSE ↓", "Misconception F1 ↑", "AUROC ↑", "Efficiency (Q to thresh) ↓"]
    
    # Filter to only the most important models for clarity
    selected_systems = ["B1_Random", "B4_MIRT", "P1_MIRT_KL", "P3_CCMIRT_CMatrix", "C4_NoSMD", "P7_Full_Neuronotes"]
    df = df[df["system"].isin(selected_systems)].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    axes      = axes.flatten()

    # Normalise efficiency for display (invert so higher=better)
    df_plot = df.copy()
    df_plot["efficiency_q_inv"] = 1.0 / df_plot["efficiency_q"].replace(0, 1)
    plot_cols = ["rmse", "f1", "auroc", "efficiency_q_inv"]
    ytitles   = ["RMSE ↓ (lower = better)", "F1 ↑", "AUROC ↑",
                  "1/EfficiencyQ ↑ (higher = fewer questions)"]

    for ax, col, title in zip(axes, plot_cols, ytitles):
        ax.set_facecolor("#16213e")
        colors = [_color_for(n) for n in df_plot["system"]]
        bars   = ax.bar(range(len(df_plot)), df_plot[col], color=colors,
                        edgecolor="white", linewidth=0.4, alpha=0.9)

        # Error bars for RMSE
        if col == "rmse":
            ax.errorbar(range(len(df_plot)),
                        df_plot["rmse"],
                        yerr=[(df_plot["rmse"] - df_plot["rmse_ci95_lo"]).values,
                              (df_plot["rmse_ci95_hi"] - df_plot["rmse"]).values],
                        fmt="none", color="white", capsize=3, linewidth=1)

        # Highlight P7 (Full Neuronotes)
        for i, name in enumerate(df_plot["system"]):
            if "P7" in name or "Full" in name:
                bars[i].set_edgecolor("#f1c40f")
                bars[i].set_linewidth(2.5)

        ax.set_xticks(range(len(df_plot)))
        ax.set_xticklabels(df_plot["system"], rotation=45, ha="right",
                           fontsize=7, color="white")
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

    fig.suptitle("Neuronotes System Comparison — All Stages",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"Saved {out_path.name}")


def plot_rmse_comparison(df_traj: pd.DataFrame, out_path: Path, top_n: int = 12):
    """RMSE-vs-question-count convergence curves for key systems."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    systems   = [c for c in df_traj.columns if c != "step"]
    # Only keep the most important models
    selected  = ["B1_Random", "B4_MIRT", "P1_MIRT_KL", "P3_CCMIRT_CMatrix", "C4_NoSMD", "P7_Full_Neuronotes"]
    selected  = [s for s in selected if s in systems]

    cmap = plt.cm.get_cmap("tab20", len(selected))
    for i, sys_name in enumerate(selected):
        if sys_name not in df_traj.columns:
            continue
        y      = df_traj[sys_name].values
        steps  = df_traj["step"].values
        lw     = 2.5 if "P7" in sys_name or "Full" in sys_name else 1.2
        ls     = "-" if "P" in sys_name else ("--" if "B" in sys_name else ":")
        ax.plot(steps, y, label=sys_name, linewidth=lw, linestyle=ls,
                color=cmap(i), alpha=0.88)

    ax.axhline(y=0.35, color="#f1c40f", linestyle="--", linewidth=1.0,
               label="RMSE threshold (0.35)", alpha=0.7)
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
    """Copy items_clean.csv → results/calibrated_item_bank.csv (no changes needed
    since dataset is already fully validated)."""
    src = DATA_DIR / "items_clean.csv"
    dst = RESULTS_DIR / "calibrated_item_bank.csv"
    if src.exists():
        pd.read_csv(src).to_csv(dst, index=False)
        print(f"Saved calibrated_item_bank.csv ({len(pd.read_csv(dst))} items)")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Neuronotes experiment runner")
    parser.add_argument("--learners", type=int, default=200,
                        help="Number of synthetic learners (default: 200)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data processing (use existing data/ folder)")
    args = parser.parse_args()

    # ---- Step 1: Data processing ----
    if not args.skip_data:
        print("\n[Step 1] Data Processing")
        data_processing.main()
    else:
        print("\n[Step 1] Skipping data processing (--skip-data flag)")

    # ---- Step 2: Simulation ----
    print("\n[Step 2] Running Experiments")
    bundles = run_all_stages(n_learners=args.learners, seed=42)

    # ---- Step 3: Save CSVs ----
    print("\n[Step 3] Saving Results")
    df_results = save_results(bundles)
    save_calibrated_item_bank()

    # ---- Step 4: Plot ----
    print("\n[Step 4] Generating Plots")
    plot_metric_comparison(df_results, RESULTS_DIR / "metric_comparison.png")

    df_traj = pd.read_csv(RESULTS_DIR / "rmse_trajectory.csv")
    plot_rmse_comparison(df_traj, RESULTS_DIR / "rmse_comparison.png")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(df_results[["system", "rmse", "f1", "auroc", "efficiency_q"]].to_string(index=False))

    # Effect size: P7 vs B4_MIRT (primary claim)
    p7 = df_results[df_results["system"].str.contains("P7")]
    b4 = df_results[df_results["system"] == "B4_MIRT"]
    if not p7.empty and not b4.empty:
        from simulation.metrics import cohens_d
        rmse_p7 = [float(p7["rmse"].values[0])]
        rmse_b4 = [float(b4["rmse"].values[0])]
        print(f"\nP7 vs B4 RMSE: {rmse_p7[0]:.4f} vs {rmse_b4[0]:.4f}")
        print(f"P7 vs B4 F1:   {float(p7['f1'].values[0]):.4f} vs {float(b4['f1'].values[0]):.4f}")

    print(f"\nAll outputs saved in: {RESULTS_DIR}")
    print("  - calibrated_item_bank.csv")
    print("  - simulation_results.csv")
    print("  - rmse_trajectory.csv")
    print("  - metric_comparison.png")
    print("  - rmse_comparison.png")


if __name__ == "__main__":
    main()
