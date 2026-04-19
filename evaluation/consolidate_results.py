#!/usr/bin/env python3
"""
consolidate_results.py – Generate plots and a markdown summary table from a
prior_eval results.json file.

Usage:
    python evaluation/consolidate_results.py path/to/results.json
    python evaluation/consolidate_results.py path/to/results.json --output-dir path/to/out/

Outputs (in <output_dir>, default <json_dir>/plots/):
    bar_rmse.png, bar_mae.png, bar_nmse.png, bar_direction_accuracy.png
    critical_difference_nmse.png
    critical_difference_direction_accuracy.png
    summary.md
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


# ── data loading ─────────────────────────────────────────────────────────────

def load_results(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_per_sample_df(data: dict) -> pd.DataFrame:
    """One row per (baseline, sample) with per-sample metrics."""
    rows = []
    for baseline, res in data["results"].items():
        for s in res.get("per_sample", []):
            rows.append({
                "baseline": baseline,
                "sample_idx": s["sample_idx"],
                "rmse": s.get("rmse"),
                "mae": s.get("mae"),
                "nmse": s.get("nmse"),
                "r2": s.get("r2"),
                "direction_accuracy": s.get("direction_accuracy"),
            })
    return pd.DataFrame(rows)


def build_summary_df(data: dict) -> pd.DataFrame:
    """One row per baseline with aggregate metrics and bootstrap CIs."""
    rows = []
    for baseline, res in data["results"].items():
        boot = res.get("bootstrap", {})
        row: dict = {
            "baseline": baseline,
            "n_samples": res.get("n_samples"),
            "rmse": res.get("rmse"),
            "mae": res.get("mae"),
            "nmse": res.get("nmse"),
            "r2": res.get("r2"),
            "direction_accuracy": res.get("direction_accuracy"),
        }
        for metric in ("rmse", "mae", "nmse", "r2", "direction_accuracy"):
            for stat in ("mean", "std", "ci_low", "ci_high"):
                row[f"{metric}_{stat}"] = boot.get(metric, {}).get(stat)
        rows.append(row)
    return pd.DataFrame(rows)


# ── bar plots ─────────────────────────────────────────────────────────────────

METRICS = [
    ("rmse",               "RMSE",               False),   # (col, label, higher_is_better)
    ("mae",                "MAE",                False),
    ("nmse",               "NMSE",               False),
    ("direction_accuracy", "Direction Accuracy",  True),
]


def _bootstrap_median_ci(values: np.ndarray, n: int = 1000, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float, float]:
    """Return (median, ci_low, ci_high) via bootstrap resampling."""
    arr = values[~np.isnan(values)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.median(arr[rng.integers(0, arr.size, size=(n, arr.size))], axis=1)
    return float(np.median(arr)), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def plot_bar_metrics(per_sample: pd.DataFrame, output_dir: Path) -> None:
    """
    One bar chart per metric showing:
      - bars    = mean  (seaborn bootstrapped 95% CI as error bars)
      - markers = median ± bootstrapped 95% CI  (diamond + whiskers overlaid)
    """
    baselines = per_sample["baseline"].unique().tolist()
    palette = sns.color_palette("tab10", n_colors=len(baselines))
    color_map = dict(zip(baselines, palette))

    for col, label, _ in METRICS:
        valid = per_sample[per_sample[col].notna()].copy()
        if valid.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(baselines) * 1.5), 5))

        # ── mean bars with bootstrapped CI ───────────────────────────────────
        sns.barplot(
            data=valid,
            x="baseline",
            y=col,
            hue="baseline",
            order=baselines,
            palette=color_map,
            capsize=0.1,
            err_kws={"linewidth": 1.5},
            legend=False,
            ax=ax,
        )

        # ── median ± bootstrap CI overlaid as diamond + whiskers ─────────────
        for i, bl in enumerate(baselines):
            vals = valid.loc[valid["baseline"] == bl, col].to_numpy(dtype=float)
            med, lo, hi = _bootstrap_median_ci(vals)
            if np.isnan(med):
                continue
            # whisker line
            ax.plot([i, i], [lo, hi], color="black", linewidth=1.5,
                    solid_capstyle="round", zorder=4)
            # caps
            for y in (lo, hi):
                ax.plot([i - 0.12, i + 0.12], [y, y], color="black",
                        linewidth=1.5, zorder=4)
            # diamond marker at median
            ax.plot(i, med, marker="D", color="white", markeredgecolor="black",
                    markersize=7, zorder=5, label="_nolegend_")

        # legend entry explaining the overlay
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="gray", linewidth=6, alpha=0.6, label="Mean (95% CI)"),
            Line2D([0], [0], marker="D", color="white", markeredgecolor="black",
                   markersize=7, linewidth=1.5, label="Median (95% CI)"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

        ax.set_title(f"{label} by Baseline", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(baselines)))
        ax.set_xticklabels(baselines, rotation=30, ha="right")
        fig.tight_layout()
        path = output_dir / f"bar_{col}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  {path.name}")


# ── critical difference diagram ───────────────────────────────────────────────

def _holm_bonferroni(p_values: dict[tuple, float], alpha: float = 0.05) -> set[tuple]:
    """Return set of (a, b) pairs that are NOT significantly different after HB correction."""
    sorted_pairs = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_pairs)
    insignificant: set[tuple] = set()
    rejected = set()
    for rank_i, (pair, p) in enumerate(sorted_pairs):
        threshold = alpha / (m - rank_i)
        if p < threshold:
            rejected.add(pair)
        else:
            # once we fail to reject we stop (Holm step-down)
            break
    for pair, _ in sorted_pairs:
        if pair not in rejected:
            insignificant.add(pair)
    return insignificant


def _find_cliques(baselines: list[str], insignificant: set[tuple]) -> list[list[str]]:
    """
    Group baselines into maximal cliques of non-significantly-different methods.
    Two baselines are connected if they are NOT significantly different.
    """
    # Build adjacency: i ~ j if not significant
    n = len(baselines)
    idx = {b: i for i, b in enumerate(baselines)}
    adj = [[False] * n for _ in range(n)]
    for a, b in insignificant:
        i, j = idx[a], idx[b]
        adj[i][j] = True
        adj[j][i] = True

    # Find maximal cliques greedily (sufficient for typical k ≤ 10)
    cliques: list[list[str]] = []
    for start in range(n):
        clique = [baselines[start]]
        for j in range(start + 1, n):
            if all(adj[idx[c]][j] for c in clique):
                clique.append(baselines[j])
        # Only keep if size > 1 (singletons are uninteresting)
        if len(clique) > 1 and not any(set(clique).issubset(set(c)) for c in cliques):
            cliques.append(clique)
    return cliques


def plot_critical_difference(per_sample: pd.DataFrame, output_dir: Path,
                              metric: str = "rmse", alpha: float = 0.05) -> None:
    """
    Draw a Critical Difference diagram.

    1. Rank baselines per sample on `metric` (lower = rank 1 for error metrics).
    2. Compute average ranks.
    3. Run pairwise Wilcoxon signed-rank tests; apply Holm-Bonferroni correction.
    4. Connect non-significantly-different groups with horizontal bars.
    """
    higher_is_better = metric == "direction_accuracy"

    pivot = per_sample.pivot_table(index="sample_idx", columns="baseline",
                                   values=metric)
    pivot = pivot.dropna(how="all")
    if pivot.empty or pivot.shape[1] < 2:
        print("  (skipping CD plot: not enough data)")
        return

    # Rank: ascending = lower error is better; descending = higher acc is better
    ascending = not higher_is_better
    ranked = pivot.rank(axis=1, method="average", ascending=ascending)
    avg_ranks = ranked.mean().sort_values()
    baselines = avg_ranks.index.tolist()

    # Pairwise Wilcoxon signed-rank tests
    p_values: dict[tuple, float] = {}
    for a, b in combinations(baselines, 2):
        col_a = pivot[a].dropna()
        col_b = pivot[b].dropna()
        common = col_a.index.intersection(col_b.index)
        if len(common) < 10:
            p_values[(a, b)] = 1.0   # too few samples to test
            continue
        diff = col_a.loc[common].values - col_b.loc[common].values
        if np.all(diff == 0):
            p_values[(a, b)] = 1.0
        else:
            _, p = wilcoxon(diff)
            p_values[(a, b)] = float(p)

    insignificant = _holm_bonferroni(p_values, alpha=alpha)
    cliques = _find_cliques(baselines, insignificant)

    # ── draw ─────────────────────────────────────────────────────────────────
    k = len(baselines)
    fig_w = max(8, k * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, 2.5 + k * 0.45))

    rank_min, rank_max = 1.0, float(k)
    y_axis = float(k) + 0.5          # y position of the rank axis line
    label_spread = 0.8               # vertical space between labels

    ax.set_xlim(rank_min - 0.6, rank_max + 0.6)
    ax.set_ylim(-0.5, y_axis + 1.2)

    # Axis line + ticks
    ax.axhline(y_axis, color="black", linewidth=1.5)
    for r in np.arange(rank_min, rank_max + 1):
        ax.plot([r, r], [y_axis, y_axis + 0.25], color="black", linewidth=1)
    ax.set_xticks(np.arange(rank_min, rank_max + 1))
    ax.set_xticklabels([str(int(r)) for r in np.arange(rank_min, rank_max + 1)])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(
        f"Average Rank on {metric.upper()}  "
        f"({'higher' if higher_is_better else 'lower'} is better → rank 1)",
        fontsize=10,
    )
    ax.yaxis.set_visible(False)
    for spine in ("left", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    # Dots on axis + vertical connectors + labels
    for i, bl in enumerate(baselines):
        r = avg_ranks[bl]
        y_label = float(k) - 1 - i * label_spread
        ax.plot(r, y_axis, "o", color="black", markersize=7, zorder=5)
        ax.plot([r, r], [y_axis - 0.15, y_label + 0.15],
                color="gray", linewidth=0.8, linestyle="--")
        ha = "left" if r <= (rank_min + rank_max) / 2 else "right"
        ax.text(r + (0.06 if ha == "left" else -0.06), y_label,
                f"{bl}  ({r:.2f})", va="center", ha=ha, fontsize=9)

    # Horizontal bars for non-significant cliques
    clique_colors = plt.cm.Set2(np.linspace(0, 0.8, max(len(cliques), 1)))
    for ci, clique in enumerate(cliques):
        ranks_in_clique = [avg_ranks[b] for b in clique]
        y_bar = -0.1 - ci * 0.22
        ax.plot([min(ranks_in_clique), max(ranks_in_clique)],
                [y_bar, y_bar], linewidth=5, color=clique_colors[ci],
                alpha=0.8, solid_capstyle="round")

    note = f"Horizontal bars: not significantly different (Wilcoxon + Holm–Bonferroni, α={alpha})"
    if len(p_values) and all(v == 1.0 for v in p_values.values()):
        note = f"Too few samples for significance testing (n<10 per pair)"
    ax.text((rank_min + rank_max) / 2, -0.45 - len(cliques) * 0.22,
            note, ha="center", fontsize=8, color="gray", style="italic")

    ax.set_title(f"Critical Difference Diagram  –  {metric.upper()}", pad=20,
                 fontsize=13)
    fig.tight_layout()
    path = output_dir / f"critical_difference_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path.name}")


# ── markdown tables ───────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 3) -> str:
    """Format a float to `decimals` places, or return '—' for missing values."""
    if val is None:
        return "—"
    if isinstance(val, float) and np.isnan(val):
        return "—"
    return f"{val:.{decimals}f}"


def _best_rows(summary: pd.DataFrame, col: str, higher_is_better: bool) -> set[int]:
    """Return the integer index positions of rows with the best value in `col`."""
    vals = summary[col].dropna()
    if vals.empty:
        return set()
    best = vals.max() if higher_is_better else vals.min()
    return set(summary.index[summary[col] == best].tolist())


def _cell(val, ci_low=None, ci_high=None, bold: bool = False, decimals: int = 3) -> str:
    """
    Format a metric cell as  `val [ci_low – ci_high]`  or just  `val`.
    Wraps in ** ** when bold=True.
    """
    v = _fmt(val, decimals)
    if ci_low is not None and ci_high is not None:
        lo, hi = _fmt(ci_low, decimals), _fmt(ci_high, decimals)
        text = f"{v} [{lo} – {hi}]"
    else:
        text = v
    return f"**{text}**" if bold else text


def build_markdown_tables(summary: pd.DataFrame) -> str:
    """
    Returns two readable markdown tables:
      1. Error metrics  – RMSE, MAE, NMSE  (lower is better, bold minimum)
      2. Direction accuracy               (higher is better, bold maximum)

    CI values are shown inline as  `mean [lo – hi]`.
    """
    blocks: list[str] = []

    # ── Table 1: Error metrics ────────────────────────────────────────────────
    best_rmse = _best_rows(summary, "rmse", higher_is_better=False)
    best_mae  = _best_rows(summary, "mae",  higher_is_better=False)
    best_nmse = _best_rows(summary, "nmse", higher_is_better=False)

    header = "| Baseline | N | RMSE [95% CI] | MAE [95% CI] | NMSE [95% CI] |"
    sep    = "|---|:---:|---|---|---|"
    rows   = [header, sep]
    for i, row in summary.iterrows():
        n = str(int(row["n_samples"])) if pd.notna(row.get("n_samples")) else "—"
        rmse_cell = _cell(row.get("rmse"), row.get("rmse_ci_low"), row.get("rmse_ci_high"),
                          bold=(i in best_rmse))
        mae_cell  = _cell(row.get("mae"),  row.get("mae_ci_low"),  row.get("mae_ci_high"),
                          bold=(i in best_mae))
        nmse_cell = _cell(row.get("nmse"), row.get("nmse_ci_low"), row.get("nmse_ci_high"),
                          bold=(i in best_nmse))
        rows.append(f"| {row['baseline']} | {n} | {rmse_cell} | {mae_cell} | {nmse_cell} |")

    blocks.append("### Error Metrics  *(lower is better — best value in bold)*\n\n"
                  + "\n".join(rows))

    # ── Table 2: Direction accuracy ────────────────────────────────────────────
    best_da = _best_rows(summary, "direction_accuracy", higher_is_better=True)

    header2 = "| Baseline | N | Direction Accuracy [95% CI] |"
    sep2    = "|---|:---:|---|"
    rows2   = [header2, sep2]
    for i, row in summary.iterrows():
        n = str(int(row["n_samples"])) if pd.notna(row.get("n_samples")) else "—"
        da_cell = _cell(row.get("direction_accuracy"),
                        row.get("direction_accuracy_ci_low"),
                        row.get("direction_accuracy_ci_high"),
                        bold=(i in best_da))
        rows2.append(f"| {row['baseline']} | {n} | {da_cell} |")

    blocks.append("### Direction Accuracy  *(higher is better — best value in bold)*\n\n"
                  + "\n".join(rows2))

    return "\n\n".join(blocks)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate prior_eval results.json into plots + markdown."
    )
    parser.add_argument("json_path", help="Path to results.json")
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: <json_dir>/plots/)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for CD diagrams (default: 0.05)",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    if not json_path.exists():
        sys.exit(f"File not found: {json_path}")

    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(json_path)
    per_sample = build_per_sample_df(data)
    summary    = build_summary_df(data)

    print(f"Loaded {len(data['results'])} baselines, "
          f"{per_sample['sample_idx'].nunique()} samples each.")
    print(f"Writing outputs to {output_dir}/\n")

    # Bar plots
    print("Bar plots:")
    plot_bar_metrics(per_sample, output_dir)

    # Critical difference diagrams (NMSE + direction accuracy)
    print("\nCritical difference diagrams:")
    plot_critical_difference(per_sample, output_dir,
                             metric="nmse", alpha=args.alpha)
    plot_critical_difference(per_sample, output_dir,
                             metric="direction_accuracy", alpha=args.alpha)

    # Markdown tables
    tscm = data.get("tscm_structure", "unknown")
    n    = data.get("n_samples", "?")
    T    = data.get("T", "?")
    md   = f"# Results: {tscm}  (n={n}, T={T})\n\n"
    md  += build_markdown_tables(summary)
    md_path = output_dir / "summary.md"
    md_path.write_text(md + "\n")
    print(f"\nMarkdown summary:\n  {md_path}\n")
    print(build_markdown_tables(summary))


if __name__ == "__main__":
    main()
