#!/usr/bin/env python3
"""
Plot Gray-Scott benchmark results produced by gs_benchmark's CSV output.

Expected columns (header written by main.cpp):
    backend,precision,scaling,ranks,size,total_time,
    communication_time,communication_fraction,updates_per_second

Multiple CSV files can be supplied. This is useful for MPI benchmarks where
each file corresponds to a different number of MPI ranks.

Usage:
    python3 plot_results.py results_1.csv results_2.csv results_4.csv

    python3 plot_results.py results/*.csv

    python3 plot_results.py results/*.csv \
        --output-dir plots --dpi 150

For each (precision, scaling) combination present in the input files, this
produces two figures:

    - updates_per_second vs. size
    - communication_fraction vs. size

Each curve represents one (backend, ranks) combination.

For weak scaling, "size" is the fixed per-rank size, so it is really a
proxy for "how much work was assigned per rank as rank count grew"; the
plot still shows what you want (does throughput per rank hold up), but
the x-axis doesn't mean the same physical thing as it does for strong
scaling.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "backend",
    "precision",
    "scaling",
    "ranks",
    "size",
    "total_time",
    "communication_time",
    "communication_fraction",
    "updates_per_second",
}


def load_results(csv_paths: list[Path]) -> pd.DataFrame:
    """Load and combine multiple benchmark CSV files."""

    frames = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"'{csv_path}' does not exist")

        if not csv_path.is_file():
            raise ValueError(f"'{csv_path}' is not a file")

        df = pd.read_csv(csv_path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"'{csv_path}' is missing expected column(s): "
                f"{sorted(missing)}. "
                "Is this a gs_benchmark output file?"
            )

        if df.empty:
            print(f"warning: '{csv_path}' has no data rows", file=sys.stderr)
            continue

        # Keep track of where every row came from. This is useful for
        # diagnosing duplicate or unexpected benchmark results.
        df["_source_file"] = csv_path.name

        frames.append(df)

        print(f"loaded {len(df)} row(s) from '{csv_path}'")

    if not frames:
        raise ValueError("none of the input files contained any data rows")

    return pd.concat(frames, ignore_index=True)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    log_y: bool,
    dpi: int,
) -> None:
    """Create one benchmark plot."""

    fig, ax = plt.subplots(figsize=(7, 5))

    # A separate curve is created for every backend/rank combination.
    for (backend, ranks), group in df.groupby(["backend", "ranks"]):
        group = group.sort_values("size")

        ax.plot(
            group["size"],
            group[metric],
            marker="o",
            label=f"{backend}, ranks={ranks}",
        )

    ax.set_xscale("log", base=2)

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4,
    )

    ax.legend(title="backend / ranks")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "csv",
        type=Path,
        nargs="+",
        help="one or more benchmark CSV files",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="directory to write PNGs into (default: ./plots)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="figure DPI (default: 150)",
    )

    args = parser.parse_args()

    try:
        df = load_results(args.csv)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("error: no benchmark data found", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots independently for each precision/scaling combination.
    for (precision, scaling), group in df.groupby(
        ["precision", "scaling"]
    ):
        ranks = sorted(group["ranks"].unique())
        backends = sorted(group["backend"].unique())

        plot_id = f"{precision}_{scaling}"

        plot_metric(
            group,
            metric="updates_per_second",
            ylabel="updates / second",
            title=(
                f"Throughput ({precision}, {scaling} scaling)"
            ),
            out_path=(
                args.output_dir
                / f"updates_per_second_{plot_id}.png"
            ),
            log_y=True,
            dpi=args.dpi,
        )

        plot_metric(
            group,
            metric="communication_fraction",
            ylabel="communication time / total time",
            title=(
                f"Communication overhead ({precision}, {scaling} scaling)"
            ),
            out_path=(
                args.output_dir
                / f"communication_fraction_{plot_id}.png"
            ),
            log_y=False,
            dpi=args.dpi,
        )

    print()
    print(f"{len(df)} total rows")
    print(f"{len(args.csv)} input file(s)")
    print(f"{df['backend'].nunique()} backend(s): "
          f"{', '.join(map(str, sorted(df['backend'].unique())))}")
    print(f"{df['ranks'].nunique()} rank count(s): "
          f"{', '.join(map(str, sorted(df['ranks'].unique())))}")
    print(f"{df['size'].nunique()} size(s)")
    print(f"plots written to '{args.output_dir}/'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())