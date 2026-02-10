"""
Community Activity Plotter
Generates per-community visualizations of r/place 2022 pixel placements.

For each community, two images are produced:
  1. Actual-color view  — each pixel drawn in the color it was placed.
  2. Heatmap view       — pixel density (placements per coordinate), showing
                          how aggressively specific areas were targeted.

Both images render the full 2000×2000 canvas with origin (0,0) at the
bottom-left corner.
"""

import polars as pl
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch PNG generation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# CONFIGURATION
# ============================================================
RPLACE_WIDTH = 2000
RPLACE_HEIGHT = 2000
OUTPUT_DIR = "botnet_plots"
DPI = 150                     # output image resolution

# Reverse color map: color_name -> RGB tuple (0-255)
# Matches the COLOR_MAP in preprocess.py
COLOR_NAME_TO_RGB = {
    "dark red": (109, 0, 26), "red": (190, 0, 57), "orange": (255, 69, 0),
    "yellow": (255, 168, 0), "pale yellow": (255, 214, 53),
    "ivory": (255, 248, 184), "dark green": (0, 163, 104),
    "green": (0, 204, 120), "light green": (126, 237, 86),
    "dark teal": (0, 117, 111), "teal": (0, 158, 170),
    "light teal": (0, 204, 192), "dark blue": (36, 80, 164),
    "blue": (54, 144, 234), "light blue": (81, 233, 244),
    "indigo": (73, 58, 193), "periwinkle": (106, 92, 255),
    "lavender": (148, 179, 255), "dark purple": (129, 30, 159),
    "purple": (180, 74, 192), "pale purple": (228, 171, 255),
    "magenta": (222, 16, 127), "pink": (255, 56, 129),
    "light pink": (255, 153, 170), "dark brown": (109, 72, 47),
    "brown": (156, 105, 38), "beige": (255, 180, 112),
    "black": (0, 0, 0), "dark gray": (81, 82, 82),
    "gray": (137, 141, 144), "light gray": (212, 215, 217),
    "white": (255, 255, 255), "unknown": (128, 128, 128),
}


def plot_community_actual_color(placements_df, community_id, community_size,
                                total_pixels, output_dir):
    """Generate an actual-color plot of a community's pixel placements.

    Each pixel is drawn in the color it was placed on the full 2000×2000
    canvas. Background is white. Origin (0,0) is at the bottom-left.

    Parameters
    ----------
    placements_df : pl.DataFrame
        Columns: x, y, color_name — all placements by this community's members.
    community_id : int
        Community identifier for the title / filename.
    community_size : int
        Number of members in the community.
    total_pixels : int
        Total pixel placements by this community.
    output_dir : str or Path
        Directory to save the output PNG.

    Returns
    -------
    Path to the saved image.
    """
    xs = placements_df["x"].to_numpy()
    ys = placements_df["y"].to_numpy()
    colors = placements_df["color_name"].to_list()

    # Build full canvas (white background)
    img = np.full((RPLACE_HEIGHT, RPLACE_WIDTH, 3), 255, dtype=np.uint8)

    # Plot each placement in its actual color
    for x, y, cname in zip(xs, ys, colors):
        if 0 <= x < RPLACE_WIDTH and 0 <= y < RPLACE_HEIGHT:
            rgb = COLOR_NAME_TO_RGB.get(cname, (128, 128, 128))
            img[y, x] = rgb

    # Create figure with origin at bottom-left
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, interpolation="nearest", origin="lower")
    ax.set_title(
        f"Community #{community_id}  —  {community_size:,} members, "
        f"{total_pixels:,} pixels",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_path = Path(output_dir) / f"community_{community_id}_color.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_community_heatmap(placements_df, community_id, community_size,
                           total_pixels, output_dir):
    """Generate a heatmap of placement density for a community.

    Brighter areas indicate more placements on the same coordinate.
    Renders the full 2000×2000 canvas. Origin (0,0) is at the bottom-left.

    Parameters
    ----------
    placements_df : pl.DataFrame
        Columns: x, y — all placements by this community's members.
    community_id : int
        Community identifier for the title / filename.
    community_size : int
        Number of members in the community.
    total_pixels : int
        Total pixel placements by this community.
    output_dir : str or Path
        Directory to save the output PNG.

    Returns
    -------
    Path to the saved image.
    """
    xs = placements_df["x"].to_numpy()
    ys = placements_df["y"].to_numpy()

    # Build full-canvas density grid
    density = np.zeros((RPLACE_HEIGHT, RPLACE_WIDTH), dtype=np.int32)
    np.add.at(density, (ys, xs), 1)

    # Create figure with origin at bottom-left
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Log scale for better visibility (many pixels have 1 placement,
    # a few hotspots have hundreds)
    norm = mcolors.LogNorm(vmin=1, vmax=max(density.max(), 2))
    # Mask zeros so background stays white
    masked = np.ma.masked_where(density == 0, density)

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest",
                   origin="lower")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Placements per pixel (log scale)", fontsize=9)

    ax.set_title(
        f"Community #{community_id} — Heatmap  —  {community_size:,} members, "
        f"{total_pixels:,} pixels",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_path = Path(output_dir) / f"community_{community_id}_heatmap.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_botnet_plots(data_path, community_df, community_scores,
                          n_top=20, output_dir=OUTPUT_DIR):
    """Generate actual-color and heatmap plots for the top botnet communities.

    Parameters
    ----------
    data_path : str
        Path to the processed parquet file.
    community_df : pl.DataFrame
        Columns: user_id_int, community_id — maps every flagged user to a community.
    community_scores : pl.DataFrame
        Per-community statistics including is_botnet, total_community_pixels, etc.
    n_top : int
        Number of top botnet communities to plot (sorted by total_community_pixels).
    output_dir : str or Path
        Directory to save images. Created if it does not exist.
    """
    import time
    start = time.perf_counter()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Select top botnet communities by total pixel count
    top_botnets = (
        community_scores
        .filter(pl.col("is_botnet"))
        .sort("total_community_pixels", descending=True)
        .head(n_top)
    )

    if top_botnets.height == 0:
        print("  > No botnet communities to plot.")
        return

    n_to_plot = top_botnets.height
    print(f"\n{'=' * 60}")
    print(f"  GENERATING BOTNET VISUALIZATIONS ({n_to_plot} communities)")
    print(f"{'=' * 60}")

    # Collect community IDs and their metadata
    comm_ids = top_botnets["community_id"].to_list()
    comm_sizes = dict(zip(
        top_botnets["community_id"].to_list(),
        top_botnets["community_size"].to_list()
    ))
    comm_pixels = dict(zip(
        top_botnets["community_id"].to_list(),
        top_botnets["total_community_pixels"].to_list()
    ))

    # Get all user_id_int values for these communities
    members_df = community_df.filter(pl.col("community_id").is_in(comm_ids))

    # Match parquet dtype for user_id_int
    lf = pl.scan_parquet(data_path)
    uid_dtype = lf.collect_schema()["user_id_int"]
    members_df = members_df.cast({"user_id_int": uid_dtype})

    # Load placements for all target community members in one pass
    print(f"  > Loading placements for {members_df.height:,} community members...")
    placements = (
        lf.join(members_df.lazy(), on="user_id_int", how="semi")
        .select(["user_id_int", "x", "y", "color_name"])
        .collect()
    )

    # Attach community_id so we can split by community
    placements = placements.join(
        members_df.select(["user_id_int", "community_id"]),
        on="user_id_int", how="left"
    )

    print(f"  > {placements.height:,} total placements loaded.")

    # Generate plots per community
    for rank, comm_id in enumerate(comm_ids, 1):
        comm_placements = placements.filter(pl.col("community_id") == comm_id)
        size = comm_sizes[comm_id]
        total_px = comm_pixels[comm_id]

        print(f"  > [{rank}/{n_to_plot}] Community #{comm_id}: "
              f"{size:,} members, {total_px:,} pixels ... ", end="", flush=True)

        color_path = plot_community_actual_color(
            comm_placements, comm_id, size, total_px, out
        )
        heat_path = plot_community_heatmap(
            comm_placements, comm_id, size, total_px, out
        )
        print(f"saved.")

    elapsed = time.perf_counter() - start
    print(f"  > All plots saved to: {out.resolve()}")
    print(f"  > Visualization complete: {elapsed:.1f}s")


def plot_all_botnets(data_path, community_df, community_scores,
                     output_dir=OUTPUT_DIR):
    """Generate combined actual-color and heatmap images for ALL botnet communities.

    Shows every pixel placed by every member of every botnet-flagged community
    on a single 2000×2000 canvas. Reveals the total footprint of botnet activity.

    Parameters
    ----------
    data_path : str
        Path to the processed parquet file.
    community_df : pl.DataFrame
        Columns: user_id_int, community_id.
    community_scores : pl.DataFrame
        Per-community statistics including is_botnet.
    output_dir : str or Path
        Directory to save images.
    """
    import time
    start = time.perf_counter()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Get all botnet community IDs
    botnet_comms = (
        community_scores
        .filter(pl.col("is_botnet"))
    )
    n_botnets = botnet_comms.height
    total_botnet_members = botnet_comms["community_size"].sum()
    total_botnet_pixels = botnet_comms["total_community_pixels"].sum()
    botnet_ids = botnet_comms["community_id"].to_list()

    print(f"\n  > Generating combined all-botnets visualization "
          f"({n_botnets} communities, {total_botnet_members:,} members)...")

    # Get all members of botnet communities
    members_df = community_df.filter(pl.col("community_id").is_in(botnet_ids))

    # Match parquet dtype
    lf = pl.scan_parquet(data_path)
    uid_dtype = lf.collect_schema()["user_id_int"]
    members_df = members_df.cast({"user_id_int": uid_dtype})

    # Load all placements in one pass
    placements = (
        lf.join(members_df.lazy(), on="user_id_int", how="semi")
        .select(["x", "y", "color_name"])
        .collect()
    )
    print(f"  > {placements.height:,} total botnet placements loaded.")

    # --- Actual-color image ---
    xs = placements["x"].to_numpy()
    ys = placements["y"].to_numpy()
    colors = placements["color_name"].to_list()

    img = np.full((RPLACE_HEIGHT, RPLACE_WIDTH, 3), 255, dtype=np.uint8)
    for x, y, cname in zip(xs, ys, colors):
        if 0 <= x < RPLACE_WIDTH and 0 <= y < RPLACE_HEIGHT:
            rgb = COLOR_NAME_TO_RGB.get(cname, (128, 128, 128))
            img[y, x] = rgb

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img, interpolation="nearest", origin="lower")
    ax.set_title(
        f"All Botnet Activity  —  {n_botnets} communities, "
        f"{total_botnet_members:,} members, {total_botnet_pixels:,} pixels",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    color_path = out / "all_botnets_color.png"
    fig.savefig(color_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Heatmap image ---
    density = np.zeros((RPLACE_HEIGHT, RPLACE_WIDTH), dtype=np.int32)
    np.add.at(density, (ys, xs), 1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    norm = mcolors.LogNorm(vmin=1, vmax=max(density.max(), 2))
    masked = np.ma.masked_where(density == 0, density)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest",
                   origin="lower")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Placements per pixel (log scale)", fontsize=9)
    ax.set_title(
        f"All Botnet Activity — Heatmap  —  {n_botnets} communities, "
        f"{total_botnet_members:,} members, {total_botnet_pixels:,} pixels",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    heat_path = out / "all_botnets_heatmap.png"
    fig.savefig(heat_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    elapsed = time.perf_counter() - start
    print(f"  > Saved: {color_path.name}, {heat_path.name}  ({elapsed:.1f}s)")


def plot_final_canvas(data_path='processed_place_data.parquet',
                      output_path='final_canvas.png'):
    """Render the final state of the r/place canvas as a PNG.

    For each (x, y) coordinate, the last-placed color (by timestamp) is used.
    Pixels with no placements remain white. Origin (0,0) is at the bottom-left.

    Parameters
    ----------
    data_path : str
        Path to the processed parquet file.
    output_path : str or Path
        Path to save the output PNG.

    Returns
    -------
    Path to the saved image.
    """
    import time
    start = time.perf_counter()
    print("--- Generating Final Canvas Snapshot ---")

    # Get the last NON-WHITE color placed at each coordinate.
    # r/place 2022 ended with a "white-out" phase where only white pixels
    # were allowed, so the absolute last color is white everywhere.
    print("  > Reading parquet and computing final pixel state "
          "(excluding white-out)...")
    final_state = (
        pl.scan_parquet(data_path)
        .filter(pl.col("color_name") != "white")
        .sort("seconds_since_start")
        .group_by(["x", "y"])
        .agg(pl.col("color_name").last().alias("final_color"))
        .collect()
    )
    print(f"  > {final_state.height:,} unique pixels with at least one "
          f"non-white placement.")

    # Build the canvas (white background)
    img = np.full((RPLACE_HEIGHT, RPLACE_WIDTH, 3), 255, dtype=np.uint8)

    xs = final_state["x"].to_numpy()
    ys = final_state["y"].to_numpy()
    color_names = final_state["final_color"].to_list()

    # Vectorized RGB lookup
    rgb_arr = np.array(
        [COLOR_NAME_TO_RGB.get(c, (128, 128, 128)) for c in color_names],
        dtype=np.uint8
    )
    valid = (xs >= 0) & (xs < RPLACE_WIDTH) & (ys >= 0) & (ys < RPLACE_HEIGHT)
    img[ys[valid], xs[valid]] = rgb_arr[valid]

    # Render with matplotlib (origin at bottom-left)
    print("  > Rendering image...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img, interpolation="nearest", origin="lower")
    ax.set_title("r/place 2022 — Final Canvas State", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out = Path(output_path)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    elapsed = time.perf_counter() - start
    print(f"  > Saved to: {out.resolve()}  ({elapsed:.1f}s)")
    return out


if __name__ == "__main__":
    plot_final_canvas()
