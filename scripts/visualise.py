#!/usr/bin/env python3
"""
Plot and analyse Gray-Scott benchmark results produced by gs_benchmark.

Expected CSV columns:
    backend,communication,precision,scaling,ranks,size,iterations,total_time,
    updates_per_second,communication_time

    backend        e.g. mpi, cuda, hip, kokkos
    communication  e.g. blocking, nonblocking
    (A legacy column named 'comm_mode' is accepted as an alias. If neither
    is present, a single dummy value "default" is used.)

`communication_fraction` is derived (communication_time / total_time) and is
set to NaN for communication types listed in --skip-comm-fraction
(default: nonblocking), because communication time cannot be measured
properly there. Those curves/panels are omitted from the plots.

Usage:
    python3 plot_results.py results/*.csv --output-dir plots --dpi 150
    python3 plot_results.py results/*.csv --reference mpi --skip-comm-fraction nonblocking

Meaning of "size"
    strong scaling: global problem size (fixed while ranks grow)
    weak scaling:   per-rank problem size (global work grows with ranks)

Assumption
    `updates_per_second` is the *global* throughput (all ranks combined).
    If your benchmark reports it per rank, pass --updates-per-rank.

Visual encoding (identical in every figure)
    colour             -> backend
    linestyle / marker -> communication

Output layout
    <output-dir>/<precision>_<scaling>/
        combined/            all communication types together
        <communication>/     one folder per communication type (e.g. blocking)
            scaling/         scaling analysis (one file per size)
    and <output-dir>/scaling_summary.csv with all derived metrics.

Files in every subset folder
    updates_per_second.png              small multiples, one panel per rank
                                        count, throughput vs. size
    communication_fraction.png          same, only if any communication time
                                        is available in the subset
    backend_comparison_vs_size.png      throughput of each backend divided by
                                        the reference backend (default: mpi),
                                        same communication type, vs. size,
                                        one panel per rank count
    backend_comparison_vs_ranks.png     same ratio vs. ranks, one panel per size
    scaling/scaling_size<N>.png         2x2 scaling analysis (vs. ranks), one
                                        file per size
Only in combined/
    communication_comparison.png        nonblocking / blocking throughput ratio

Scaling metrics
    Baseline = smallest rank count present for each
    (backend, communication, precision, scaling, size) series (ideally 1).
    With R = ranks / baseline_ranks and T = throughput / baseline throughput:
        speedup = T          efficiency = T / R
    Karp-Flatt (strong, baseline = 1 rank):
        e = (1/S - 1/p) / (1 - 1/p)
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, ScalarFormatter


REQUIRED_COLUMNS = {
    "backend",
    "precision",
    "scaling",
    "ranks",
    "size",
    "total_time",
    "communication_time",
    "updates_per_second",
}

NUMERIC_COLUMNS = [
    "ranks",
    "size",
    "iterations",
    "total_time",
    "communication_time",
    "updates_per_second",
]

KEY = ["backend", "communication", "precision", "scaling", "ranks", "size"]
SERIES_KEY = ["backend", "communication", "precision", "scaling", "size"]

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


# ----------------------------------------------------------------------------
# Loading and cleaning
# ----------------------------------------------------------------------------
def load_results(csv_paths: list[Path]) -> pd.DataFrame:
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
                f"{sorted(missing)}. Is this a gs_benchmark output file?"
            )

        if "communication" not in df.columns:
            if "comm_mode" in df.columns:
                df = df.rename(columns={"comm_mode": "communication"})
            else:
                warn(f"'{csv_path}' has no 'communication' column; using 'default'")
                df["communication"] = "default"

        if df.empty:
            warn(f"'{csv_path}' has no data rows")
            continue

        df["_source_file"] = csv_path.name
        frames.append(df)
        print(f"loaded {len(df)} row(s) from '{csv_path}'")

    if not frames:
        raise ValueError("none of the input files contained any data rows")

    return pd.concat(frames, ignore_index=True)


def clean_and_aggregate(
    df: pd.DataFrame, how: str, updates_per_rank: bool, skip_comm_fraction: list[str]
) -> pd.DataFrame:
    df = df.copy()

    if "iterations" not in df.columns:
        df["iterations"] = np.nan

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("backend", "communication", "precision", "scaling"):
        df[col] = df[col].astype(str).str.strip().str.lower()

    required_numeric = ["ranks", "size", "total_time", "communication_time", "updates_per_second"]
    bad = df[required_numeric].isna().any(axis=1)
    bad |= (df["total_time"] <= 0) | (df["ranks"] <= 0) | (df["updates_per_second"] <= 0)
    if bad.any():
        warn(f"dropping {int(bad.sum())} row(s) with missing or non-positive values")
        df = df[~bad]

    if df.empty:
        raise ValueError("no valid rows remain after cleaning")

    df["ranks"] = df["ranks"].astype(int)
    df["size"] = df["size"].astype(int)

    counts = df.groupby(KEY).size()
    n_dup = int((counts > 1).sum())
    if n_dup:
        warn(f"{n_dup} configuration(s) have repeated measurements; combining them with '{how}'")

    agg = (
        df.groupby(KEY)
        .agg(
            iterations=("iterations", how),
            total_time=("total_time", how),
            communication_time=("communication_time", how),
            updates_per_second=("updates_per_second", how),
            n_samples=("total_time", "size"),
        )
        .reset_index()
    )

    agg["communication_fraction"] = agg["communication_time"] / agg["total_time"]
    skip = [s.strip().lower() for s in skip_comm_fraction]
    agg.loc[agg["communication"].isin(skip), "communication_fraction"] = np.nan

    agg["updates_global"] = agg["updates_per_second"] * (agg["ranks"] if updates_per_rank else 1)

    iters = agg["iterations"].where(agg["iterations"] > 0)
    agg["time_per_iteration"] = agg["total_time"] / iters.fillna(1.0)

    return agg


# ----------------------------------------------------------------------------
# Scaling analysis
# ----------------------------------------------------------------------------
def amdahl_serial_fraction(ranks: np.ndarray, speedup: np.ndarray) -> float | None:
    """Least-squares fit of f in 1/S = f + (1 - f)/p. Needs a 1-rank baseline."""
    mask = ranks > 1
    if not mask.any():
        return None

    p = ranks[mask].astype(float)
    x = 1.0 - 1.0 / p
    y = 1.0 / speedup[mask] - 1.0 / p
    f = float(np.sum(x * y) / np.sum(x * x))
    return min(max(f, 0.0), 1.0)


def add_scaling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    for _, g in df.groupby(SERIES_KEY):
        g = g.sort_values("ranks").copy()
        base = g.iloc[0]
        r0 = int(base["ranks"])

        rel_ranks = g["ranks"] / r0
        ratio = g["updates_global"] / base["updates_global"]

        g["baseline_ranks"] = r0
        g["speedup"] = ratio
        g["ideal_speedup"] = rel_ranks
        g["efficiency"] = ratio / rel_ranks

        g["karp_flatt"] = np.nan
        g["amdahl_f"] = np.nan
        if g["scaling"].iloc[0].startswith("strong") and r0 == 1:
            p = g["ranks"].to_numpy(dtype=float)
            s = g["speedup"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                kf = (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)
            kf[p <= 1] = np.nan
            g["karp_flatt"] = kf
            f = amdahl_serial_fraction(g["ranks"].to_numpy(), s)
            if f is not None:
                g["amdahl_f"] = f

        out.append(g)

    return pd.concat(out, ignore_index=True)


def relative_to_reference(group: pd.DataFrame, reference: str, tag: str) -> pd.DataFrame | None:
    """Throughput of every backend divided by the reference backend's throughput.

    Matched on (communication, size, ranks). Returns a long table with columns
    backend, communication, size, ranks, ratio (reference itself excluded),
    or None if nothing can be computed.
    """
    if reference not in set(group["backend"].unique()):
        warn(f"{tag}: reference backend '{reference}' not present; skipping backend comparison")
        return None

    piv = group.pivot_table(
        index=["communication", "size", "ranks"], columns="backend", values="updates_global"
    )
    others = [b for b in piv.columns if b != reference]
    if not others:
        return None

    rows = []
    for b in others:
        r = (piv[b] / piv[reference]).dropna().rename("ratio").reset_index()
        r["backend"] = b
        rows.append(r)
    ratio = pd.concat(rows, ignore_index=True)
    if ratio.empty:
        warn(f"{tag}: no overlapping configurations with '{reference}'")
        return None
    return ratio


# ----------------------------------------------------------------------------
# Style: one global mapping so every figure encodes things the same way
# ----------------------------------------------------------------------------
class Style:
    def __init__(self, df: pd.DataFrame):
        self.backends = sorted(df["backend"].unique())
        self.communications = sorted(
            df["communication"].unique(), key=lambda m: (m != "blocking", m)
        )
        cmap = plt.get_cmap("tab10")
        self.color = {b: cmap(i % 10) for i, b in enumerate(self.backends)}
        self.linestyle = {m: LINESTYLES[i % len(LINESTYLES)] for i, m in enumerate(self.communications)}
        self.marker = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(self.communications)}

    def kw(self, backend: str, comm: str) -> dict:
        return dict(
            color=self.color[backend],
            linestyle=self.linestyle[comm],
            marker=self.marker[comm],
            markersize=4,
        )

    def legend_handles(self, backends, comms) -> list:
        """Compact legend: one entry per backend (colour) + one per communication (style)."""
        handles = [Line2D([], [], color=self.color[b], linewidth=2, label=b) for b in backends]
        if len(comms) > 1:
            handles += [
                Line2D([], [], color="0.3", linestyle=self.linestyle[m], marker=self.marker[m],
                       markersize=4, label=m)
                for m in comms
            ]
        return handles


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------
def set_rank_axis(ax, ranks) -> None:
    ranks = sorted(set(int(r) for r in ranks))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("MPI ranks")


def size_label(scaling: str) -> str:
    return "per-rank size" if scaling.startswith("weak") else "global size"


def make_grid(n: int, max_cols: int = 3, cell=(4.4, 3.6), **kwargs):
    ncols = min(n, max_cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False, **kwargs
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def finish(fig, handles, out_path: Path, dpi: int, title: str | None = None, ncol: int = 6) -> None:
    n_rows = int(np.ceil(len(handles) / ncol)) if handles else 0
    bottom = 0.02 + 0.045 * n_rows
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=9, frameon=False)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, bottom, 1, 0.95 if title else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_vs_size(
    group: pd.DataFrame, metric: str, ylabel: str, title: str, xlabel: str,
    out_path: Path, log_y: bool, dpi: int, style: Style,
) -> bool:
    """Small multiples: one panel per rank count, colour=backend, style=communication."""

    group = group.dropna(subset=[metric])
    if group.empty:
        return False

    rank_list = sorted(group["ranks"].unique())
    fig, axes = make_grid(len(rank_list), sharey=True)

    for ax, r in zip(axes, rank_list):
        sub = group[group["ranks"] == r]
        for (backend, comm), g in sub.groupby(["backend", "communication"]):
            g = g.sort_values("size")
            ax.plot(g["size"], g[metric], **style.kw(backend, comm))

        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(f"{r} rank{'s' if r != 1 else ''}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    for ax in axes:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(ylabel)
    if not log_y:
        for ax in axes:
            ax.set_ylim(bottom=0)

    handles = style.legend_handles(
        sorted(group["backend"].unique()), sorted(group["communication"].unique())
    )
    finish(fig, handles, out_path, dpi, title=title)
    return True


def plot_scaling(
    group: pd.DataFrame, size: int, precision: str, scaling: str, subset: str,
    out_path: Path, dpi: int, style: Style,
) -> bool:
    """2x2 scaling figure for one (precision, scaling, subset, size)."""

    strong = scaling.startswith("strong")

    series = [
        (key, g.sort_values("ranks"))
        for key, g in group.groupby(["backend", "communication"])
        if g["ranks"].nunique() >= 2
    ]
    if not series:
        warn(f"no series with >= 2 rank counts for {precision}/{scaling}/{subset}/size={size}; skipping")
        return False

    baselines = {int(g["baseline_ranks"].iloc[0]) for _, g in series}
    if baselines != {1}:
        warn(
            f"{precision}/{scaling}/{subset}/size={size}: baseline rank count(s) "
            f"{sorted(baselines)} (not 1); metrics are relative to the smallest rank "
            "count per series"
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax_speed, ax_eff, ax_time, ax_comm = axes.ravel()

    all_ranks: set[int] = set()
    handles = []

    for (backend, comm), g in series:
        kw = style.kw(backend, comm)
        ranks = g["ranks"].to_numpy()
        all_ranks.update(ranks.tolist())

        label = f"{backend}, {comm}"
        f = g["amdahl_f"].iloc[0]
        if strong and not np.isnan(f):
            label += f" (f≈{f:.3f})"

        (line,) = ax_speed.plot(ranks, g["speedup"], label=label, **kw)
        handles.append(line)

        ax_eff.plot(ranks, g["efficiency"], **kw)
        ax_time.plot(ranks, g["time_per_iteration"], **kw)
        if g["communication_fraction"].notna().any():
            ax_comm.plot(ranks, g["communication_fraction"], **kw)

    r_min, r_max = min(all_ranks), max(all_ranks)
    r0 = min(baselines)
    (ideal,) = ax_speed.plot(
        [r_min, r_max], [r_min / r0, r_max / r0], color="black", linestyle=":", label="ideal"
    )
    handles.append(ideal)
    ax_speed.set_yscale("log", base=2)
    ax_speed.set_ylabel("speedup" if strong else "scaled speedup")
    ax_speed.set_title("Speedup")

    ax_eff.axhline(1.0, color="black", linestyle=":")
    ax_eff.set_ylim(0, max(1.1, float(group["efficiency"].max()) * 1.05))
    ax_eff.set_ylabel("parallel efficiency")
    ax_eff.set_title("Parallel efficiency")

    ax_time.set_yscale("log")
    ax_time.set_ylabel("time per iteration [s]")
    ax_time.set_title("Time per iteration")

    ax_comm.set_ylabel("communication time / total time")
    ax_comm.set_title("Communication fraction")
    if group["communication_fraction"].notna().any():
        ax_comm.set_ylim(bottom=0)
    else:
        ax_comm.text(0.5, 0.5, "not measured", ha="center", va="center",
                     transform=ax_comm.transAxes, color="0.4")
        ax_comm.set_yticks([])

    for ax in axes.ravel():
        set_rank_axis(ax, all_ranks)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    title = (f"{scaling.capitalize()} scaling ({precision}, {subset}), "
             f"{size_label(scaling)} = {size}")
    finish(fig, handles, out_path, dpi, title=title, ncol=3)
    return True


def plot_backend_comparison(
    ratio: pd.DataFrame, x: str, reference: str, precision: str, scaling: str,
    subset: str, out_path: Path, dpi: int, style: Style,
) -> bool:
    """Throughput relative to `reference` (same communication type).

    x == "ranks": one panel per size,       ratio vs. ranks
    x == "size":  one panel per rank count, ratio vs. size
    """

    panel_col = "size" if x == "ranks" else "ranks"
    panels = sorted(ratio[panel_col].unique())
    fig, axes = make_grid(len(panels), sharey=True)

    for ax, p in zip(axes, panels):
        sub = ratio[ratio[panel_col] == p]
        for (backend, comm), g in sub.groupby(["backend", "communication"]):
            g = g.sort_values(x)
            ax.plot(g[x], g["ratio"], **style.kw(backend, comm))
        ax.axhline(1.0, color="black", linestyle=":")

        if x == "ranks":
            set_rank_axis(ax, sub["ranks"].unique())
            ax.set_title(f"{size_label(scaling)} = {p}", fontsize=10)
        else:
            ax.set_xscale("log", base=2)
            ax.set_xlabel(size_label(scaling))
            ax.set_title(f"{p} rank{'s' if p != 1 else ''}", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    for ax in axes:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(f"throughput / {reference} throughput")

    handles = style.legend_handles(
        sorted(ratio["backend"].unique()), sorted(ratio["communication"].unique())
    )
    finish(fig, handles, out_path, dpi,
           title=f"Backends relative to {reference} ({precision}, {scaling} scaling, {subset})")
    return True


def plot_communication_comparison(
    group: pd.DataFrame, precision: str, scaling: str, out_path: Path, dpi: int,
    style: Style, num: str = "nonblocking", den: str = "blocking",
) -> bool:
    """Throughput ratio num/den vs ranks; one panel per size, one line per backend."""

    if not {num, den} <= set(group["communication"].unique()):
        return False

    piv = group.pivot_table(
        index=["backend", "size", "ranks"], columns="communication", values="updates_global"
    ).dropna(subset=[num, den])
    if piv.empty:
        return False
    piv["ratio"] = piv[num] / piv[den]
    piv = piv.reset_index()

    sizes = sorted(piv["size"].unique())
    fig, axes = make_grid(len(sizes), sharey=True)

    for ax, size in zip(axes, sizes):
        sub = piv[piv["size"] == size]
        for backend, g in sub.groupby("backend"):
            g = g.sort_values("ranks")
            ax.plot(g["ranks"], g["ratio"], color=style.color[backend], marker="o", markersize=4)
        ax.axhline(1.0, color="black", linestyle=":")
        set_rank_axis(ax, sub["ranks"].unique())
        ax.set_title(f"{size_label(scaling)} = {size}", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    for ax in axes:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(f"throughput {num} / {den}")

    handles = style.legend_handles(sorted(piv["backend"].unique()), [])
    finish(fig, handles, out_path, dpi,
           title=f"{num.capitalize()} vs {den} ({precision}, {scaling} scaling)")
    return True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, nargs="+", help="one or more benchmark CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"),
                        help="directory to write outputs into (default: ./plots)")
    parser.add_argument("--dpi", type=int, default=150, help="figure DPI (default: 150)")
    parser.add_argument("--aggregate", choices=["median", "mean", "min"], default="median",
                        help="how to combine repeated measurements (default: median)")
    parser.add_argument("--updates-per-rank", action="store_true",
                        help="treat updates_per_second as a per-rank value (default: global)")
    parser.add_argument("--reference", default="mpi",
                        help="reference backend for the backend comparison (default: mpi)")
    parser.add_argument("--skip-comm-fraction", nargs="*", default=["nonblocking"],
                        help="communication types whose communication fraction is not "
                             "plotted (default: nonblocking)")
    args = parser.parse_args()
    reference = args.reference.strip().lower()

    try:
        raw = load_results(args.csv)
        df = clean_and_aggregate(raw, args.aggregate, args.updates_per_rank,
                                 args.skip_comm_fraction)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    df = add_scaling_metrics(df)
    style = Style(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for (precision, scaling), group in df.groupby(["precision", "scaling"]):
        xlabel = size_label(scaling)

        subsets = [("combined", group)] + [
            (c, group[group["communication"] == c]) for c in style.communications
            if (group["communication"] == c).any()
        ]

        for subset, sub in subsets:
            out_dir = args.output_dir / f"{precision}_{scaling}" / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{precision}, {scaling} scaling, {subset}"

            plot_vs_size(
                sub, "updates_per_second", "updates / second",
                f"Throughput ({tag})", xlabel,
                out_dir / "updates_per_second.png", log_y=True, dpi=args.dpi, style=style,
            )
            plot_vs_size(
                sub, "communication_fraction", "communication time / total time",
                f"Communication overhead ({tag})", xlabel,
                out_dir / "communication_fraction.png", log_y=False, dpi=args.dpi, style=style,
            )

            ratio = relative_to_reference(sub, reference, f"{precision}/{scaling}/{subset}")
            if ratio is not None:
                for x in ("size", "ranks"):
                    plot_backend_comparison(
                        ratio, x, reference, precision, scaling, subset,
                        out_dir / f"backend_comparison_vs_{x}.png", args.dpi, style,
                    )

            for size, size_group in sub.groupby("size"):
                plot_scaling(size_group, int(size), precision, scaling, subset,
                             out_dir / "scaling" / f"scaling_size{size}.png",
                             args.dpi, style)

            if subset == "combined":
                plot_communication_comparison(
                    sub, precision, scaling,
                    out_dir / "communication_comparison.png", args.dpi, style,
                )

    # Summary table.
    cols = [
        "backend", "communication", "precision", "scaling", "size", "ranks", "baseline_ranks",
        "time_per_iteration", "updates_global", "speedup", "ideal_speedup",
        "efficiency", "communication_fraction", "karp_flatt", "amdahl_f", "n_samples",
    ]
    summary = df.sort_values(SERIES_KEY + ["ranks"])[cols]
    summary_path = args.output_dir / "scaling_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    print()
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", 200):
        print(
            summary[
                ["backend", "communication", "precision", "scaling", "size", "ranks",
                 "speedup", "efficiency", "communication_fraction", "karp_flatt"]
            ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )

    print()
    print(f"{len(df)} configuration(s) after aggregation ({len(raw)} raw rows)")
    print(f"{len(args.csv)} input file(s)")
    print(f"{df['backend'].nunique()} backend(s): {', '.join(style.backends)}")
    print(f"{df['communication'].nunique()} communication type(s): {', '.join(style.communications)}")
    print(f"{df['ranks'].nunique()} rank count(s): {', '.join(map(str, sorted(df['ranks'].unique())))}")
    print(f"{df['size'].nunique()} size(s)")
    print(f"outputs written to '{args.output_dir}/'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())