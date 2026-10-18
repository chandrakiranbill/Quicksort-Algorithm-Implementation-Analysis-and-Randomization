"""
benchmark.py
============

Empirical comparison of deterministic vs. randomized Quicksort.

For every (input distribution, input size) pair the script:
  1. builds the input array,
  2. times each algorithm over several trials on fresh copies of that array
     (the MEDIAN wall-clock time is reported to suppress noise),
  3. separately counts element comparisons and maximum recursion depth,
  4. writes everything to ``results/benchmark_results.csv`` and
     ``results/benchmark_results.md``, and
  5. draws the figures in ``results/`` (requires matplotlib).

Usage (from the repository root):
    python src/benchmark.py                  # full run (~1-3 minutes)
    python src/benchmark.py --quick          # small sizes, a few seconds
    python src/benchmark.py --sizes 1000 2000 --trials 3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quicksort import SortStats, quicksort, randomized_quicksort  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

DEFAULT_SIZES = [500, 1000, 2000, 4000, 8000]
DEFAULT_TRIALS = 5
SEED = 2026


# --------------------------------------------------------------------------- #
#  Input distributions                                                        #
# --------------------------------------------------------------------------- #
def make_random(n: int, rng: random.Random) -> List[int]:
    return [rng.randint(0, 10 * n) for _ in range(n)]


def make_sorted(n: int, rng: random.Random) -> List[int]:
    return list(range(n))


def make_reverse(n: int, rng: random.Random) -> List[int]:
    return list(range(n, 0, -1))


def make_few_unique(n: int, rng: random.Random) -> List[int]:
    """Many duplicate keys: only 10 distinct values."""
    return [rng.randint(0, 9) for _ in range(n)]


DISTRIBUTIONS: Dict[str, Callable[[int, random.Random], List[int]]] = {
    "random": make_random,
    "sorted": make_sorted,
    "reverse": make_reverse,
    "few_unique": make_few_unique,
}

ALGORITHMS = {
    "deterministic": lambda a, s=None: quicksort(a, s),
    "randomized": lambda a, s=None: randomized_quicksort(a, s),
}


# --------------------------------------------------------------------------- #
#  Measurement                                                                #
# --------------------------------------------------------------------------- #
def time_once(sort_fn, data: List[int]) -> float:
    arr = data.copy()
    start = time.perf_counter()
    sort_fn(arr)
    elapsed = time.perf_counter() - start
    assert arr == sorted(data), "sort produced wrong output"
    return elapsed


def run(sizes: List[int], trials: int) -> List[dict]:
    rng = random.Random(SEED)
    random.seed(SEED)  # makes randomized pivots reproducible run-to-run
    rows: List[dict] = []
    for dist_name, make in DISTRIBUTIONS.items():
        for n in sizes:
            data = make(n, rng)
            for algo_name, sort_fn in ALGORITHMS.items():
                times = [time_once(sort_fn, data) for _ in range(trials)]
                stats = SortStats()
                sort_fn(data.copy(), stats)
                row = {
                    "distribution": dist_name,
                    "n": n,
                    "algorithm": algo_name,
                    "median_time_ms": round(statistics.median(times) * 1000, 3),
                    "min_time_ms": round(min(times) * 1000, 3),
                    "max_time_ms": round(max(times) * 1000, 3),
                    "comparisons": stats.comparisons,
                    "max_depth": stats.max_depth,
                    "n_log2_n": round(n * math.log2(n)),
                    "n_sq_over_2": n * (n - 1) // 2,
                }
                rows.append(row)
                print(f"{dist_name:>10} n={n:>6} {algo_name:>13}: "
                      f"{row['median_time_ms']:>10.2f} ms  "
                      f"{row['comparisons']:>11,} comparisons", flush=True)
    return rows


def expected_comparisons(n: int) -> float:
    """Exact expected comparisons of randomized Quicksort: 2(n+1)H_n - 4n,
    which is asymptotically 2n ln n ~= 1.39 n log2 n."""
    harmonic = sum(1.0 / k for k in range(1, n + 1))
    return 2 * (n + 1) * harmonic - 4 * n


def read_csv(path: str) -> List[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("n", "comparisons", "max_depth", "n_log2_n", "n_sq_over_2"):
            r[k] = int(r[k])
        for k in ("median_time_ms", "min_time_ms", "max_time_ms"):
            r[k] = float(r[k])
    return rows


# --------------------------------------------------------------------------- #
#  Output                                                                     #
# --------------------------------------------------------------------------- #
def write_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[dict], path: str) -> None:
    lines = ["# Benchmark results", "",
             "Median wall-clock time (ms) and comparison counts.", ""]
    for dist in DISTRIBUTIONS:
        lines += [f"## {dist}", "",
                  "| n | Deterministic (ms) | Randomized (ms) | "
                  "Det. comparisons | Rand. comparisons | Speed-up |",
                  "|---:|---:|---:|---:|---:|---:|"]
        sizes = sorted({r["n"] for r in rows if r["distribution"] == dist})
        for n in sizes:
            d = next(r for r in rows if r["distribution"] == dist
                     and r["n"] == n and r["algorithm"] == "deterministic")
            z = next(r for r in rows if r["distribution"] == dist
                     and r["n"] == n and r["algorithm"] == "randomized")
            speed = d["median_time_ms"] / z["median_time_ms"]
            lines.append(f"| {n:,} | {d['median_time_ms']:.2f} | "
                         f"{z['median_time_ms']:.2f} | {d['comparisons']:,} | "
                         f"{z['comparisons']:,} | {speed:.2f}x |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot(rows: List[dict], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except ImportError:
        print("matplotlib not installed - skipping plots.")
        return

    colors = {"deterministic": "#2a78d6", "randomized": "#eb6834"}
    labels = {"deterministic": "Deterministic (last-element pivot)",
              "randomized": "Randomized pivot"}
    titles = {"random": "Random input", "sorted": "Sorted input",
              "reverse": "Reverse-sorted input",
              "few_unique": "Few unique keys (10 values)"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": muted, "axes.labelcolor": ink,
        "xtick.color": muted, "ytick.color": muted,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })

    for metric, ylabel, fname in (
        ("median_time_ms", "Median time (ms, log scale)", "fig_runtime.png"),
        ("comparisons", "Comparisons (log scale)", "fig_comparisons.png"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4), sharex=True)
        for ax, dist in zip(axes.flat, DISTRIBUTIONS):
            for algo in ALGORITHMS:
                pts = sorted((r["n"], r[metric]) for r in rows
                             if r["distribution"] == dist
                             and r["algorithm"] == algo)
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=colors[algo], lw=2, marker="o",
                        ms=5, mec="#fcfcfb", mew=1.5, label=labels[algo])
            if metric == "comparisons":
                xs = sorted({r["n"] for r in rows})
                ax.plot(xs, [expected_comparisons(n) for n in xs], ls="--",
                        lw=1.2, color=muted,
                        label=r"Theory (avg): $2(n+1)H_n-4n$")
                ax.plot(xs, [n * (n - 1) / 2 for n in xs], ls=":", lw=1.4,
                        color=muted, label=r"Theory: $n(n-1)/2$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            sizes = sorted({r["n"] for r in rows})
            ax.set_xticks(sizes)
            ax.set_xticklabels([f"{n:,}" for n in sizes])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_title(titles[dist], loc="left", color=ink, fontsize=10)
            ax.grid(True, which="major", color=grid, lw=0.8)
            ax.set_axisbelow(True)
        for ax in axes[1]:
            ax.set_xlabel("Input size n (log scale)")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        handles, lbls = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center",
                   ncol=len(lbls), frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close(fig)
        print("wrote", os.path.join(out_dir, fname))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--quick", action="store_true",
                    help="small sizes and 3 trials, for a fast smoke test")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw figures from the existing CSV without re-timing")
    args = ap.parse_args()
    if args.plot_only:
        plot(read_csv(os.path.join(RESULTS_DIR, "benchmark_results.csv")),
             RESULTS_DIR)
        return
    if args.quick:
        args.sizes, args.trials = [250, 500, 1000], 3

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Python {sys.version.split()[0]} | sizes={args.sizes} "
          f"| trials={args.trials}\n")
    rows = run(args.sizes, args.trials)
    write_csv(rows, os.path.join(RESULTS_DIR, "benchmark_results.csv"))
    write_markdown(rows, os.path.join(RESULTS_DIR, "benchmark_results.md"))
    if not args.no_plots:
        plot(rows, RESULTS_DIR)
    print("\nResults written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
