#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from otf2.reader import Reader


# ==================================================
# Config
# ==================================================

@dataclass(frozen=True)
class RenderOptions:
    # Filtering
    min_duration_ms: float = 1.0  # events shorter than this are dropped

    # Drawing performance
    chunk_size: int = 3000

    # Lane view thinning
    max_lane_rows: int = 40  # max rows shown in the top lane view

    # Depth view
    max_depth: int = 20

    # Label rendering (depth view)
    label_font_size: int = 5
    label_visible_ratio: float = 0.05  # draw label only if visible part >= this ratio of full range

    # Legend
    legend_font_size: int = 8
    legend_wrap_width: int = 120
    legend_line_space: float = 0.020  # axes fraction
    legend_y_start: float = 0.94
    legend_y_end: float = 0.04

    # Page layout (A4 portrait)
    page_size_inches: Tuple[float, float] = (8.27, 11.69)


# ==================================================
# Helpers: CLI parsing
# ==================================================

def parse_ranges(tokens: Sequence[str]) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for t in tokens:
        m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$", t)
        if not m:
            raise ValueError(f"Invalid range token: {t!r}. Expected 'tmin,tmax' (e.g., 0,120).")
        tmin = float(m.group(1))
        tmax = float(m.group(2))
        if tmax < tmin:
            raise ValueError(f"Invalid range (tmax < tmin): {tmin},{tmax}")
        ranges.append((tmin, tmax))
    return ranges


def parse_lane_spec(s: str) -> str:
    m = re.match(r"^\s*r(\d+)\s*:\s*t(\d+)\s*$", s)
    if not m:
        raise ValueError(f"Invalid --focus lane: {s!r}. Expected 'r<rank>:t<thread>' (e.g., r0:t0).")
    return f"r{int(m.group(1))}:t{int(m.group(2))}"


# ==================================================
# Helpers: labels / colors
# ==================================================

def core_label_from_region(region_name: str) -> str:
    region_name = re.sub(r"\[with.*?\]$", "", region_name).strip()
    matches = list(re.finditer(r"([A-Za-z_]\w*)\s*\(", region_name))
    return matches[-1].group(1) if matches else region_name


def stable_hash_rgb(key: str) -> Tuple[float, float, float]:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return ((h >> 16) & 255) / 255.0, ((h >> 8) & 255) / 255.0, (h & 255) / 255.0


def to_pastel(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    r, g, b = rgb
    return ((r + 1) / 2, (g + 1) / 2, (b + 1) / 2)


def region_color_rule(region_name: str) -> Tuple[float, float, float]:
    if re.search(r"\bmain\b", region_name):
        return (0.35, 0.35, 0.35)

    if re.match(r"^MPI_", region_name):
        return (1.00, 0.20, 0.20)
    if re.match(r"^PMPI_", region_name):
        return (1.00, 0.50, 0.30)

    if re.search(r"(read|write|open|close|fread|fwrite)", region_name):
        return (0.20, 0.40, 1.00)

    if re.match(r"^(omp_|GOMP_)", region_name):
        return (1.00, 0.60, 0.00)

    if re.search(r"(init|start|final|setup)", region_name, re.IGNORECASE):
        return (0.50, 0.50, 0.50)

    if region_name.startswith("Foam::"):
        return (0.00, 0.45, 0.70)

    if region_name.startswith("std::"):
        return (0.55, 0.55, 0.55)

    return to_pastel(stable_hash_rgb(region_name))


def build_region_color_map(region_names: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    return {r: region_color_rule(r) for r in region_names}


# ==================================================
# OTF2: location -> rank, thread
# ==================================================

def get_rank_id_from_location(loc) -> int:
    """
    OTF2 Location -> LocationGroup ID (MPI rank相当)
    Try multiple attribute names to absorb binding differences.
    """
    for attr in ("group", "location_group", "parent"):
        if hasattr(loc, attr):
            g = getattr(loc, attr)
            if g is not None and hasattr(g, "_ref"):
                return int(g._ref)
    raise AttributeError("Cannot find LocationGroup (rank) from location object")


# ==================================================
# Sampling helpers (top lane thinning)
# ==================================================

def sample_evenly(items: Sequence[str], max_n: int) -> List[str]:
    """
    Keep at most max_n items, evenly spaced, always include first and last.
    """
    n = len(items)
    if max_n <= 0 or n == 0:
        return []
    if n <= max_n:
        return list(items)
    if max_n == 1:
        return [items[0]]

    idx = []
    for i in range(max_n):
        j = int(round(i * (n - 1) / (max_n - 1)))
        idx.append(j)

    seen = set()
    out = []
    for j in idx:
        if j not in seen:
            out.append(items[j])
            seen.add(j)

    # Ensure first/last
    if out and out[0] != items[0]:
        out[0] = items[0]
    if out and out[-1] != items[-1]:
        out[-1] = items[-1]
    return out


# ==================================================
# Legend pagination helpers
# ==================================================

def paginate_legend_regions(region_names: Sequence[str], opt: RenderOptions) -> List[List[str]]:
    """
    Split region list into pages so that wrapped lines fit within y_start..y_end.
    """
    capacity_lines = int((opt.legend_y_start - opt.legend_y_end) / opt.legend_line_space)
    capacity_lines = max(1, capacity_lines)

    pages: List[List[str]] = []
    cur: List[str] = []
    used = 0

    for region in region_names:
        wrapped = textwrap.fill(region, width=opt.legend_wrap_width)
        n_lines = len(wrapped.split("\n"))

        if cur and (used + n_lines > capacity_lines):
            pages.append(cur)
            cur = []
            used = 0

        cur.append(region)
        used += n_lines

        if used >= capacity_lines:
            pages.append(cur)
            cur = []
            used = 0

    if cur:
        pages.append(cur)
    return pages


# ==================================================
# OTF2 load / preprocess
# ==================================================

def load_otf2_intervals_rank_thread(trace_path: str) -> Tuple[pd.DataFrame, int]:
    print("[INFO] Loading OTF2 trace (this may take a while)...")

    reader = Reader(trace_path)
    resolution = reader.timer_resolution

    # (rank, loc_id) -> thread_index (0..)
    thread_index: Dict[Tuple[int, int], int] = {}
    thread_count_per_rank: Dict[int, int] = {}

    # lane -> stack[(enter_event, depth)]
    enter_stack: Dict[str, List[Tuple[object, int]]] = {}

    rows: List[dict] = []

    for loc, ev in reader.events:
        loc_id = int(loc._ref)
        rank_id = get_rank_id_from_location(loc)

        key = (rank_id, loc_id)
        if key not in thread_index:
            next_tid = thread_count_per_rank.get(rank_id, 0)
            thread_index[key] = next_tid
            thread_count_per_rank[rank_id] = next_tid + 1

        tid = thread_index[key]
        lane = f"r{rank_id}:t{tid}"

        enter_stack.setdefault(lane, [])
        etype = type(ev).__name__

        if etype == "Enter":
            enter_stack[lane].append((ev, len(enter_stack[lane])))

        elif etype == "Leave" and enter_stack[lane]:
            en, depth = enter_stack[lane].pop()
            rows.append(dict(
                lane=lane,
                rank=rank_id,
                thread=tid,
                location=loc_id,
                region=en.region.name,
                start=en.time,
                end=ev.time,
                depth=depth,
            ))

    reader.close()
    df = pd.DataFrame(rows)
    return df, resolution


def preprocess_intervals(df: pd.DataFrame, resolution: int, opt: RenderOptions) -> pd.DataFrame:
    df = df.copy()
    df["duration"] = df["end"] - df["start"]

    min_ticks = int((opt.min_duration_ms / 1000.0) * resolution)
    df = df[df["duration"] >= min_ticks]

    t0 = df["start"].min()
    df["start_s"] = (df["start"] - t0) / resolution
    df["end_s"] = (df["end"] - t0) / resolution
    df["duration_s"] = df["duration"] / resolution
    return df


def make_default_ranges_from_df(df: pd.DataFrame, n_split: int) -> List[Tuple[float, float]]:
    """
    Default ranges:
      - Always include overview page [tmin, tmax]
      - n_split == 1: overview only
      - n_split >= 2: overview + n_split equal subranges
      - n_split <= 0 is invalid (checked in main)
    """
    tmin = float(df["start_s"].min())
    tmax = float(df["end_s"].max())
    if tmax <= tmin:
        return [(tmin, tmin + 1e-6)]

    ranges: List[Tuple[float, float]] = [(tmin, tmax)]

    if n_split <= 1:
        return ranges

    width = tmax - tmin
    q = width / n_split

    for i in range(n_split):
        a = tmin + i * q
        b = tmin + (i + 1) * q if i < n_split - 1 else tmax
        ranges.append((a, b))

    return ranges


def pick_default_focus_lane(df: pd.DataFrame) -> str:
    """
    Default focus: min-rank thread0 if exists, otherwise the smallest (rank,thread).
    """
    lanes = df[["rank", "thread", "lane"]].drop_duplicates().sort_values(["rank", "thread"])
    if lanes.empty:
        return "r0:t0"
    min_rank = int(lanes["rank"].min())
    cand = lanes[(lanes["rank"] == min_rank) & (lanes["thread"] == 0)]
    if not cand.empty:
        return str(cand.iloc[0]["lane"])
    return str(lanes.iloc[0]["lane"])


# ==================================================
# Rendering
# ==================================================

def render_timeline_page(
    df: pd.DataFrame,
    region_color: Dict[str, Tuple[float, float, float]],
    tmin: float,
    tmax: float,
    page_num: int,
    opt: RenderOptions,
    focus_lane: str,
    focus_max_depth_global: int,
):
    print(f"[INFO] Generating page {page_num}: Range {tmin} – {tmax} [sec]")

    fig = plt.figure(figsize=opt.page_size_inches)
    fig.subplots_adjust(left=0.17, right=0.9, top=0.92, bottom=0.05)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 0.5], hspace=0.3)

    ax_lane = fig.add_subplot(gs[0])
    ax_depth = fig.add_subplot(gs[1])
    ax_footer = fig.add_subplot(gs[2])
    ax_footer.axis("off")

    # -----------------------------
    # 1) Lane view: all lanes (rank+thread) with thinning
    # -----------------------------
    df_lane_all = df[(df["end_s"] >= tmin) & (df["start_s"] <= tmax)].sort_values("start_s")
    print(f"   Visible events (all lanes, before thinning): {len(df_lane_all):,}")

    lanes_visible = (
        df_lane_all[["lane", "rank", "thread"]]
        .drop_duplicates()
        .sort_values(["rank", "thread"], ascending=[True, True])  # ascending for sampling
    )["lane"].tolist()

    lanes_sampled = sample_evenly(lanes_visible, opt.max_lane_rows)
    df_lane = df_lane_all[df_lane_all["lane"].isin(set(lanes_sampled))]

    lanes_display = (
        df_lane[["lane", "rank", "thread"]]
        .drop_duplicates()
        .sort_values(["rank", "thread"], ascending=[False, False])
    )["lane"].tolist()

    print(f"   Lanes shown (top view): {len(lanes_display)} / {len(lanes_visible)}  (max={opt.max_lane_rows})")
    print(f"   Visible events (top view, after thinning): {len(df_lane):,}")

    lane_height, lane_gap = 10, 4
    y_lane = lambda lane: lanes_display.index(lane) * (lane_height + lane_gap)

    for start_i in range(0, len(df_lane), opt.chunk_size):
        sub = df_lane.iloc[start_i:start_i + opt.chunk_size]
        rects, colors = [], []
        for _, r in sub.iterrows():
            rects.append(Rectangle((r.start_s, y_lane(r.lane)), r.duration_s, lane_height))
            colors.append(region_color[r.region])
        ax_lane.add_collection(PatchCollection(rects, facecolor=colors, edgecolor="none"))

    n_lanes = max(1, len(lanes_display))
    font_lane = max(4, min(10, 180 / n_lanes))

    # Title: only annotate when thinning actually happened
    if len(lanes_visible) > opt.max_lane_rows:
        lane_title = f"All Rank/Thread Lanes (showing up to {opt.max_lane_rows} lanes)"
    else:
        lane_title = "All Rank/Thread Lanes"

    ax_lane.set_title(lane_title, fontsize=12)
    ax_lane.set_xlim(tmin, tmax)
    ax_lane.set_xlabel("time [s]")
    ax_lane.set_yticks([y_lane(l) + lane_height / 2 for l in lanes_display])
    ax_lane.set_yticklabels(lanes_display, fontsize=font_lane)
    ax_lane.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax_lane.ticklabel_format(style="plain", axis="x")

    if lanes_display:
        ax_lane.set_ylim(-1, (len(lanes_display) - 1) * (lane_height + lane_gap) + lane_height + 2)

    # -----------------------------
    # 2) Depth view: focused lane
    # -----------------------------
    df_focus = df[(df["lane"] == focus_lane) &
                  (df["end_s"] >= tmin) &
                  (df["start_s"] <= tmax)].copy()

    print(f"   Visible events (focus {focus_lane}): {len(df_focus):,}")

    df_focus.loc[df_focus["depth"] > opt.max_depth, "depth"] = opt.max_depth
    df_focus = df_focus.sort_values("start_s")

    max_depth = min(int(focus_max_depth_global), opt.max_depth)

    depth_height, depth_gap = 8, 2
    y_depth = lambda d: (max_depth - d) * (depth_height + depth_gap)

    for start_i in range(0, len(df_focus), opt.chunk_size):
        sub = df_focus.iloc[start_i:start_i + opt.chunk_size]
        rects, colors = [], []
        for _, r in sub.iterrows():
            rects.append(Rectangle((r.start_s, y_depth(r.depth)), r.duration_s, depth_height))
            colors.append(region_color[r.region])
        ax_depth.add_collection(PatchCollection(rects, facecolor=colors, edgecolor="none"))

    n_depth = max_depth + 1
    font_depth = max(4, min(10, 150 / n_depth))

    ax_depth.set_title(f"Focused Lane Timeline: {focus_lane}", fontsize=12)
    ax_depth.set_xlim(tmin, tmax)
    ax_depth.set_xlabel("time [s]")
    ax_depth.set_yticks([y_depth(d) + depth_height / 2 for d in range(max_depth + 1)])
    ax_depth.set_yticklabels([f"depth {d}" for d in range(max_depth + 1)], fontsize=font_depth)
    ax_depth.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax_depth.ticklabel_format(style="plain", axis="x")

    if max_depth >= 0:
        ax_depth.set_ylim(-1, y_depth(0) + depth_height + 2)

    # -----------------------------
    # Header texts
    # -----------------------------
    fig.text(0.02, 0.97, f"Range: {tmin:.6g} – {tmax:.6g}  [sec]", fontsize=14, ha="left")
    fig.text(0.98, 0.97, f"Page {page_num}", fontsize=12, ha="right")

    # -----------------------------
    # Labels inside depth bars
    # -----------------------------
    full_range = tmax - tmin
    visible_threshold = full_range * opt.label_visible_ratio
    font_size = opt.label_font_size

    for _, r in df_focus.iterrows():
        start_s = r.start_s
        end_s = r.start_s + r.duration_s
        vs, ve = max(start_s, tmin), min(end_s, tmax)
        visible_w = max(0.0, ve - vs)
        if visible_w < visible_threshold:
            continue

        func = core_label_from_region(r.region)
        dur = r.duration_s
        dstr = f"{dur:.2f}s" if dur >= 1 else f"{dur * 1000:.1f}ms"
        label = f"{func} ({dstr})"

        x0_px = ax_depth.transData.transform((vs, 0))[0]
        x1_px = ax_depth.transData.transform((ve, 0))[0]
        bar_px = x1_px - x0_px
        max_chars = max(5, int(bar_px / (font_size * 2.0)))

        wrapped = textwrap.fill(label, width=max_chars)
        lines = wrapped.split("\n")
        if len(lines) > 3:
            t = lines[2]
            lines = [lines[0], lines[1], (t[:-1] + "…") if len(t) > 1 else "…"]

        lane_top = y_depth(r.depth) + depth_height
        step = depth_height / (len(lines) + 1)
        x_text = vs + full_range * 0.005

        for i, line in enumerate(lines):
            y_text = lane_top - (i + 1) * step
            ax_depth.text(
                x_text, y_text, line,
                fontsize=font_size, ha="left", va="center",
                color="black", clip_on=True
            )

    return fig


def render_legend_page(
    region_subset: Sequence[str],
    region_color: Dict[str, Tuple[float, float, float]],
    opt: RenderOptions,
    legend_page_index: int,
    legend_page_total: int,
):
    fig = plt.figure(figsize=opt.page_size_inches)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.04)

    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    title = "Legend (All Regions)" if legend_page_total == 1 else f"Legend (All Regions)  {legend_page_index}/{legend_page_total}"
    fig.text(0.5, 0.975, title, ha="center", fontsize=16)

    y = opt.legend_y_start
    for region in region_subset:
        wrapped = textwrap.fill(region, width=opt.legend_wrap_width)
        lines = wrapped.split("\n")
        for i, line in enumerate(lines):
            prefix = "■ " if i == 0 else "   "
            ax.text(
                0.06, y,
                prefix + line,
                color=region_color[region],
                fontsize=opt.legend_font_size,
                ha="left", va="top",
                transform=ax.transAxes
            )
            y -= opt.legend_line_space

    return fig


def render_legend_pages(
    region_names: Sequence[str],
    region_color: Dict[str, Tuple[float, float, float]],
    opt: RenderOptions,
) -> List[plt.Figure]:
    pages = paginate_legend_regions(region_names, opt)
    figs: List[plt.Figure] = []
    total = len(pages)
    for i, subset in enumerate(pages, 1):
        figs.append(render_legend_page(subset, region_color, opt, i, total))
    return figs


# ==================================================
# main
# ==================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate multi-range timeline PDF from OTF2 trace (rank+thread lanes)."
    )
    p.add_argument("trace", help="Path to traces.otf2")
    p.add_argument("-o", "--output", default="timeline_multi.pdf", help="Output PDF path")

    p.add_argument(
        "--ranges",
        nargs="+",
        default=None,
        help="List of time ranges as 'tmin,tmax' (sec). Example: --ranges 0,120 0,10 0,5",
    )

    p.add_argument(
        "--auto-split",
        type=int,
        default=10,
        help=(
            "Default range split count N. "
            "N=1: overview only. "
            "N>=2: overview + N equal subranges. "
            "0 is not allowed."
        ),
    )

    p.add_argument(
        "--focus",
        default=None,
        help="Focused lane for depth view (default: min-rank thread0 if exists). Format: r<rank>:t<thread> e.g., r0:t0",
    )

    p.add_argument("--min-duration-ms", type=float, default=1.0, help="Drop events shorter than this (ms)")
    p.add_argument("--max-depth", type=int, default=20, help="Max depth shown in depth view")
    p.add_argument("--label-visible-ratio", type=float, default=0.05, help="Min visible ratio to draw labels in depth view")
    p.add_argument("--chunk-size", type=int, default=3000, help="Rectangles per batch to draw (performance tuning)")

    p.add_argument("--max-lane-rows", type=int, default=40, help="Max number of rows in top lane view (thinning applied)")

    p.add_argument("--legend-wrap-width", type=int, default=120, help="Wrap width for legend region names")
    p.add_argument("--legend-line-space", type=float, default=0.020, help="Legend line spacing in axes fraction")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.auto_split <= 0:
        raise SystemExit("--auto-split must be >= 1 (N=1 overview only, N>=2 adds N equal splits).")

    opt = RenderOptions(
        min_duration_ms=args.min_duration_ms,
        max_depth=args.max_depth,
        label_visible_ratio=args.label_visible_ratio,
        chunk_size=args.chunk_size,
        max_lane_rows=args.max_lane_rows,
        legend_wrap_width=args.legend_wrap_width,
        legend_line_space=args.legend_line_space,
    )

    df_raw, resolution = load_otf2_intervals_rank_thread(args.trace)
    df = preprocess_intervals(df_raw, resolution, opt)

    if df.empty:
        raise RuntimeError("No events after filtering. Try lowering --min-duration-ms or check the trace.")

    # Ranges: user-specified or auto-generated
    if args.ranges is not None:
        ranges = parse_ranges(args.ranges)
    else:
        ranges = make_default_ranges_from_df(df, args.auto_split)

    # Region colors
    region_names = sorted(df["region"].unique())
    region_color = build_region_color_map(region_names)

    # Focus lane
    if args.focus is not None:
        focus_lane = parse_lane_spec(args.focus)
    else:
        focus_lane = pick_default_focus_lane(df)

    if focus_lane not in set(df["lane"].unique()):
        focus_lane = pick_default_focus_lane(df)

    focus_max_depth_global = int(df[df["lane"] == focus_lane]["depth"].max())
    t_global_min = float(df["start_s"].min())
    t_global_max = float(df["end_s"].max())

    print("======== Trace Summary ========")
    print(f"Total events: {len(df):,}")
    print(f"Ranks: {df['rank'].nunique()}")
    print(f"Lanes (rank+thread): {df['lane'].nunique()}")
    print(f"Unique regions: {len(region_names)}")
    print(f"Global time range: {t_global_min:.6g} – {t_global_max:.6g} [sec]")
    print(f"Focus lane: {focus_lane} (max depth global: {focus_max_depth_global})")
    print(f"Top view max rows: {opt.max_lane_rows}")
    if args.ranges is None:
        if args.auto_split == 1:
            print("Default ranges: overview only (auto-split=1)")
        else:
            print(f"Default ranges: overview + {args.auto_split} equal splits (auto-split={args.auto_split})")
    else:
        print("User ranges: (overrides --auto-split)")
    for i, (a, b) in enumerate(ranges, 1):
        print(f"  {i}: {a:.6g} – {b:.6g}")
    print("================================")

    with PdfPages(args.output) as pdf:
        page_num = 1

        # Timeline pages
        for (tmin, tmax) in ranges:
            fig = render_timeline_page(
                df, region_color, tmin, tmax, page_num, opt,
                focus_lane=focus_lane,
                focus_max_depth_global=focus_max_depth_global,
            )
            pdf.savefig(fig, metadata={"Title": f"Range {tmin}-{tmax} [sec]"})
            plt.close(fig)
            page_num += 1

        # Legend pages (auto-paginated)
        legend_figs = render_legend_pages(region_names, region_color, opt)
        print(f"[INFO] Generating legend pages: {len(legend_figs)} page(s)")

        for i, fig in enumerate(legend_figs, 1):
            pdf.savefig(fig, metadata={"Title": f"Legend {i}/{len(legend_figs)}"})
            plt.close(fig)

    print(f"[INFO] PDF written: {args.output}")


if __name__ == "__main__":
    main()
