"""
Botnet Analysis via Leiden Community Detection
Analyzes r/place 2022 data to identify coordinated bot accounts using a
six-phase pipeline: feature extraction, behavioral flagging, co-occurrence
graph construction, Leiden community detection, community scoring, and
final classification.
"""

import polars as pl
import numpy as np
import igraph as ig
import leidenalg
from scipy.sparse import coo_matrix
import time
from com_activity_plotter import generate_botnet_plots, plot_all_botnets

# ============================================================
# CONFIGURATION - PHASE 1: Feature Extraction
# ============================================================
DATA_FILE_PATH = 'processed_place_data.parquet'
RPLACE_WIDTH = 2000
RPLACE_HEIGHT = 2000
MIN_PLACEMENTS = 10          # Users with fewer placements are excluded

# ============================================================
# CONFIGURATION - PHASE 2: Behavioral Flagging Thresholds
# ============================================================
COOLDOWN_5M_PCT_THRESHOLD = 0.5      # >50% of intervals near 5m cooldown
LOW_STD_INTERVAL_THRESHOLD = 15.0    # Seconds -- mechanical regularity
CONTINUOUS_HOURS_THRESHOLD = 24.0    # No human clicks for 24h straight
SMALL_AREA_MAX_BBOX = 9              # 3x3 bounding box
PRINTER_ADJACENT_THRESHOLD = 0.7     # 70%+ consecutive pixels are adjacent
PRINTER_AXIS_THRESHOLD = 0.6         # 60%+ movement along single axis
HIGH_VOLUME_PERCENTILE = 0.97        # Top 3% by placement count

# ============================================================
# CONFIGURATION - PHASE 3: Graph Construction
# ============================================================
TILE_SIZE = 50                       # Pixels per tile side (50x50)
WINDOW_SECONDS = 300                 # 5-minute temporal bins
MIN_CO_OCCURRENCE = 3                # Minimum shared bins for an edge

# ============================================================
# CONFIGURATION - PHASE 4: Leiden Community Detection
# ============================================================
LEIDEN_RESOLUTION = 0.02             # CPM resolution (min internal density)

# ============================================================
# CONFIGURATION - PHASE 5: Community Scoring
# ============================================================
MIN_COMMUNITY_SIZE = 3               # Ignore singleton/pair communities
COMMUNITY_TILE_PRESENCE_PCT = 0.25   # 25% of members must touch a tile
BOTNET_FLAG_DENSITY_THRESHOLD = 1.5  # Avg flags per member
BOTNET_COOLDOWN_THRESHOLD = 0.4      # Avg cooldown precision
BOTNET_TEMPORAL_STD_THRESHOLD = 7200.0   # 2-hour std = tight synchronization
BOTNET_MIN_SIZE_FOR_COOLDOWN = 5
BOTNET_MIN_SIZE_FOR_UNIFORMITY = 10
BOTNET_UNIFORMITY_STD_THRESHOLD = 10.0   # Low std of member std_intervals

# ============================================================
# CONFIGURATION - PHASE 6: Final Classification
# ============================================================
SCORE_WEIGHTS = {
    "flag_cooldown": 3.0,
    "flag_low_interval_std": 3.0,
    "flag_24h_active": 2.0,
    "flag_small_area": 1.0,
    "flag_single_color": 0.5,
    "flag_printer": 2.0,
    "flag_high_volume": 0.5,
}
SCORE_BOTNET_COMMUNITY = 2.0
SCORE_LARGE_BOTNET_COMMUNITY = 4.0
LARGE_COMMUNITY_SIZE = 20
HIGH_CONFIDENCE_SCORE = 5.0
PROBABLE_BOT_SCORE = 3.0
TOP_COMMUNITIES_TO_REPORT = 5

# ============================================================
# Derived constants (do not modify)
# ============================================================
NUM_TILE_COLS = RPLACE_WIDTH // TILE_SIZE
NUM_TILE_ROWS = RPLACE_HEIGHT // TILE_SIZE


def tile_id_to_ranges(tile_id):
    """Convert a tile_id back to pixel coordinate ranges."""
    tx = tile_id // NUM_TILE_ROWS
    ty = tile_id % NUM_TILE_ROWS
    return (tx * TILE_SIZE, tx * TILE_SIZE + TILE_SIZE - 1,
            ty * TILE_SIZE, ty * TILE_SIZE + TILE_SIZE - 1)


# ============================================================
# PHASE 1: Per-User Feature Extraction
# ============================================================
def phase1_extract_features(data_path):
    """Extract behavioral features for every user with >= MIN_PLACEMENTS.

    Returns (user_features DataFrame, total_unique_users int).
    total_unique_users counts ALL users in the dataset (before filtering)
    and is used as the denominator for bot-percentage reporting.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Per-User Feature Extraction")
    print("=" * 60)
    start = time.perf_counter()

    lf = pl.scan_parquet(data_path)

    # Count ALL unique users in the dataset (before any filtering)
    print("  > Counting total unique users in dataset...")
    total_unique_users = (
        lf.select(pl.col("user_id_int").n_unique()).collect().item()
    )
    print(f"  > Total unique users in dataset: {total_unique_users:,}")

    # ------------------------------------------------------------------
    # Main group_by: timing, spatial, color, printer features
    # ------------------------------------------------------------------
    print("  > Computing main feature aggregations...")
    user_features = (
        lf.group_by("user_id_int")
        .agg([
            # ---- Timing ----
            pl.len().alias("total_placements"),

            (pl.col("seconds_since_start").max()
             - pl.col("seconds_since_start").min())
            .alias("active_span_s"),

            pl.col("seconds_since_start").sort().diff().drop_nulls()
            .median().alias("median_interval"),

            pl.col("seconds_since_start").sort().diff().drop_nulls()
            .std().alias("std_interval"),

            # Cooldown precision: fraction of intervals near 5-min cooldown
            (pl.col("seconds_since_start").sort().diff()
             .is_between(295, 310).sum().cast(pl.Float64)
             / (pl.len() - 1).cast(pl.Float64))
            .alias("pct_near_cooldown_5m"),

            # ---- Spatial ----
            pl.col("x").cast(pl.Float64).std().alias("x_std"),
            pl.col("y").cast(pl.Float64).std().alias("y_std"),

            ((pl.col("x").max() - pl.col("x").min() + 1).cast(pl.Int64)
             * (pl.col("y").max() - pl.col("y").min() + 1).cast(pl.Int64))
            .alias("bounding_box_area"),

            pl.struct(["x", "y"]).n_unique().alias("unique_pixels"),

            # ---- Color ----
            pl.col("color_name").n_unique().alias("unique_colors"),

            # ---- Printer Detection ----
            # pct_adjacent: fraction of consecutive pairs within Manhattan dist 2
            # more robust at finding printers by not requiring the movement to be in the same direction
            (
                ((pl.col("x").sort_by("seconds_since_start").diff().abs()
                  + pl.col("y").sort_by("seconds_since_start").diff().abs())
                 <= 2).sum().cast(pl.Float64)
                / (pl.len() - 1).cast(pl.Float64)
            ).alias("pct_adjacent"),

            # pct_single_axis_movement: either dx==0 or dy==0
            (
                ((pl.col("x").sort_by("seconds_since_start").diff() == 0)
                 | (pl.col("y").sort_by("seconds_since_start").diff() == 0))
                .sum().cast(pl.Float64)
                / (pl.len() - 1).cast(pl.Float64)
            ).alias("pct_single_axis_movement"),

            # sweep_score_x: avg run length of same-sign horizontal movement
            # Higher = longer unbroken sweeps = more printer-like
            (
                pl.col("x").sort_by("seconds_since_start").diff()
                .ne(0).sum().cast(pl.Float64)
                / ((pl.col("x").sort_by("seconds_since_start").diff()
                    .sign().diff().abs() == 2).sum() + 1).cast(pl.Float64)
            ).alias("sweep_score_x"),

            # sweep_score_y: same for vertical axis
            (
                pl.col("y").sort_by("seconds_since_start").diff()
                .ne(0).sum().cast(pl.Float64)
                / ((pl.col("y").sort_by("seconds_since_start").diff()
                    .sign().diff().abs() == 2).sum() + 1).cast(pl.Float64)
            ).alias("sweep_score_y"),

            # ---- used in Phase 5  ----
            pl.col("seconds_since_start").mean()
            .cast(pl.Float64).alias("mean_placement_time"),
        ])
        .filter(pl.col("total_placements") >= MIN_PLACEMENTS)
        .collect()
    )

    # Combine sweep scores into single metric (dominant axis)
    if "sweep_score_x" in user_features.columns and "sweep_score_y" in user_features.columns:
        user_features = user_features.with_columns(
            pl.max_horizontal("sweep_score_x", "sweep_score_y")
            .alias("sweep_score")
        ).drop(["sweep_score_x", "sweep_score_y"])

    # Fill any NaN / null from edge cases (single-element groups, etc.)
    float_cols = [c for c in user_features.columns
                  if user_features[c].dtype in (pl.Float64, pl.Float32)]
    user_features = user_features.with_columns([
        pl.col(c).fill_nan(0.0).fill_null(0.0) for c in float_cols
    ])

    print(f"  > Main features computed for {user_features.height:,} users "
          f"(>= {MIN_PLACEMENTS} placements)")

    # ------------------------------------------------------------------
    # Dominant color percentage (separate double group-by)
    # ------------------------------------------------------------------
    print("  > Computing dominant color percentages...")
    dominant_color = (
        lf.group_by(["user_id_int", "color_name"])
        .agg(pl.len().alias("color_count"))
        .group_by("user_id_int")
        .agg([
            pl.col("color_count").max().alias("max_color_count"),
            pl.col("color_count").sum().alias("total_color_count"),
        ])
        .with_columns(
            (pl.col("max_color_count").cast(pl.Float64)
             / pl.col("total_color_count").cast(pl.Float64))
            .alias("dominant_color_pct")
        )
        .select(["user_id_int", "dominant_color_pct"])
        .collect()
    )
    user_features = user_features.join(dominant_color, on="user_id_int", how="left")
    user_features = user_features.with_columns(
        pl.col("dominant_color_pct").fill_null(0.0)
    )

    # ------------------------------------------------------------------
    # Max continuous hours (consecutive-hour streak method)
    # Defines "continuous activity" as at least one placement per hour.
    # Uses the difference-from-sequence trick: for consecutive hour_ids
    # like 5,6,7 minus cumulative counts 1,2,3 the result is constant (4).
    # A gap in hours produces a new constant, identifying each streak.
    # Much more efficient than sessionization: operates on unique (user, hour)
    # pairs rather than the full 160M-row dataset.
    # ------------------------------------------------------------------
    print("  > Computing max continuous activity hours...")
    max_hours = (
        lf.select([
            "user_id_int",
            (pl.col("seconds_since_start") / 3600).cast(pl.Int32).alias("hour_id"),
        ])
        .unique()
        .sort(["user_id_int", "hour_id"])
        .with_columns(
            (pl.col("hour_id")
             - pl.col("hour_id").cum_count().over("user_id_int"))
            .alias("streak_id")
        )
        .group_by(["user_id_int", "streak_id"])
        .agg(pl.len().alias("streak_length"))
        .group_by("user_id_int")
        .agg(pl.col("streak_length").max().cast(pl.Float64)
             .alias("max_continuous_hours"))
        .collect()
    )
    user_features = user_features.join(max_hours, on="user_id_int", how="left")
    user_features = user_features.with_columns(
        pl.col("max_continuous_hours").fill_null(0.0)
    )

    # ------------------------------------------------------------------
    # Placements per hour derived from active span and total placements
    # ------------------------------------------------------------------
    user_features = user_features.with_columns(
        pl.when(pl.col("active_span_s") > 0)
        .then(
            pl.col("total_placements").cast(pl.Float64)
            / (pl.col("active_span_s").cast(pl.Float64) / 3600.0)
        )
        .otherwise(0.0)
        .alias("placements_per_hour")
    )

    elapsed = time.perf_counter() - start
    print(f"  > Phase 1 complete: {user_features.height:,} users, "
          f"{user_features.width} features, {elapsed:.1f}s")
    return user_features, total_unique_users


# ============================================================
# PHASE 2: Behavioral Flagging
# ============================================================
def phase2_flag_users(user_features):
    """Apply behavioral thresholds and flag suspicious users."""
    print("\n" + "=" * 60)
    print("PHASE 2: Behavioral Flagging")
    print("=" * 60)
    start = time.perf_counter()

    p97 = user_features["total_placements"].quantile(HIGH_VOLUME_PERCENTILE)
    print(f"  > High-volume threshold (P{HIGH_VOLUME_PERCENTILE*100:.0f}): "
          f"{p97:.0f} placements")

    # All users already have >= MIN_PLACEMENTS (filtered in Phase 1)
    user_features = user_features.with_columns([
        (pl.col("pct_near_cooldown_5m") > COOLDOWN_5M_PCT_THRESHOLD)
        .alias("flag_cooldown"),
        # bots have a low standard deviation of interval
        (
            (pl.col("std_interval") < LOW_STD_INTERVAL_THRESHOLD)
            & (pl.col("std_interval") > 0)
        ).alias("flag_low_interval_std"),

        # bots need no sleep
        (pl.col("max_continuous_hours") > CONTINUOUS_HOURS_THRESHOLD)
        .alias("flag_24h_active"),

        # bots only place pixels within a small area
        (pl.col("bounding_box_area") <= SMALL_AREA_MAX_BBOX)
        .alias("flag_small_area"),

        # bots only place a single color
        (pl.col("unique_colors") == 1)
        .alias("flag_single_color"),

        # bots are printers
        (
            (pl.col("pct_adjacent") > PRINTER_ADJACENT_THRESHOLD)
            & (pl.col("pct_single_axis_movement") > PRINTER_AXIS_THRESHOLD)
        ).alias("flag_printer"),
        # bots place a lot of pixels
        (pl.col("total_placements") > p97).alias("flag_high_volume")
    ])

    # Compute flag_count (sum of boolean flags as UInt8)
    flag_cols = [c for c in user_features.columns if c.startswith("flag_")]
    user_features = user_features.with_columns(
        sum(pl.col(c).cast(pl.UInt8) for c in flag_cols).alias("flag_count")
    )

    # Report
    total = user_features.height
    for fc in flag_cols:
        count = user_features.filter(pl.col(fc).cast(pl.Boolean)).height
        print(f"  > {fc}: {count:,} users ({count/total*100:.2f}%)")

    flagged = user_features.filter(pl.col("flag_count") > 0)
    flagged_ids = set(flagged["user_id_int"].to_list())

    elapsed = time.perf_counter() - start
    print(f"  > Total flagged: {len(flagged_ids):,} / {total:,} "
          f"({len(flagged_ids)/total*100:.2f}%), {elapsed:.1f}s")
    return user_features, flagged_ids


# ============================================================
# PHASE 3: Co-occurrence Graph Construction
# ============================================================
def phase3_build_graph(data_path, flagged_ids):
    """Build a weighted co-occurrence graph among flagged users.

    Iterates over spatial-temporal bins individually, generating pairwise
    co-occurrence observations with numpy and accumulating them into a
    scipy sparse matrix in batches. This avoids the O(N^2) intermediate
    memory explosion of a full self-join. Bins exceeding the 16 GB memory
    threshold are randomly subsampled to keep peak memory bounded.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Co-occurrence Graph Construction")
    print("=" * 60)
    start = time.perf_counter()

    lf = pl.scan_parquet(data_path)

    # Match the dtype of user_id_int from the parquet file for join compatability 
    # and to avoid type errors
    uid_dtype = lf.collect_schema()["user_id_int"]
    flagged_df = pl.DataFrame({
        "user_id_int": list(flagged_ids)
    }).cast({"user_id_int": uid_dtype})

    # Add tile and window IDs, filter to flagged users only
    filtered_lf = (
        lf.join(flagged_df.lazy(), on="user_id_int", how="semi")
        .with_columns([
            ((pl.col("x").cast(pl.Int32) // TILE_SIZE) * NUM_TILE_ROWS
             + (pl.col("y").cast(pl.Int32) // TILE_SIZE))
            .cast(pl.Int32).alias("tile_id"),

            (pl.col("seconds_since_start") // WINDOW_SECONDS)
            .cast(pl.Int32).alias("window_id"),
        ])
    )

    # Collect only the columns we need (minimizes memory usage)
    print("  > Scanning parquet for flagged user placements...")
    filtered_df = (
        filtered_lf
        .select(["user_id_int", "x", "y", "tile_id", "window_id"])
        .collect()
    )
    print(f"  > {filtered_df.height:,} placements from {len(flagged_ids):,} "
          f"flagged users")

    # ---- User tile sets (for Phase 5 spatial coherence) ----
    print("  > Computing per-user tile sets...")
    user_tiles = (
        filtered_df.select(["user_id_int", "tile_id"]).unique()
        .group_by("user_id_int")
        .agg(pl.col("tile_id"))
    )

    # ---- Bin boundary diagnostic ---- Used to detect frequencies of edge cases
    # near_x = ((pl.col("x") % TILE_SIZE < 2) | (pl.col("x") % TILE_SIZE >= TILE_SIZE - 2))
    # near_y = ((pl.col("y") % TILE_SIZE < 2) | (pl.col("y") % TILE_SIZE >= TILE_SIZE - 2))
    # near_boundary_count = filtered_df.filter(near_x | near_y).height
    # pct_boundary = near_boundary_count / max(filtered_df.height, 1) * 100
    # print(f"  > Boundary diagnostic: {pct_boundary:.1f}% of placements within "
    #       f"2px of a tile edge (upper bound on split co-occurrences)")

    # ---- Build unique user-per-bin table ----
    # Deduplicate: a user appearing multiple times in the same bin counts once
    # This should not remove many users, but is a cheap operation to ensure there are no duplicates with inflated co-occurrence counts
    print("  > Building unique user-bin assignments...")
    user_bin_df = (
        filtered_df.select(["user_id_int", "tile_id", "window_id"])
        .unique()
    )
    print(f"  > {user_bin_df.height:,} unique (user, tile, window) assignments")

    # Filter out bins with fewer than 2 users (no pairs possible)
    valid_bins = (
        user_bin_df.group_by(["tile_id", "window_id"])
        .agg(pl.len().alias("n_users"))
        .filter(pl.col("n_users") >= 2)
    )
    print(f"  > {valid_bins.height:,} bins with 2+ flagged users")

    # Keep only user-bin rows that belong to valid bins
    user_bin_df = user_bin_df.join(
        valid_bins.select(["tile_id", "window_id"]),
        on=["tile_id", "window_id"],
        how="semi"
    )

    # Free filtered_df -- no longer needed, reclaim memory for the join
    del filtered_df

    # ---- Map user IDs to contiguous indices for sparse matrix ----
    print("  > Mapping user IDs to contiguous indices...")
    unique_users_sorted = user_bin_df["user_id_int"].unique().sort()
    n_graph_users = unique_users_sorted.len()
    idx_map_df = pl.DataFrame({
        "user_id_int": unique_users_sorted,
        "_idx": range(n_graph_users),
    }).cast({"_idx": pl.Int32})
    user_bin_df = user_bin_df.join(idx_map_df, on="user_id_int", how="left")

    # Numpy lookup array: contiguous index -> original user_id_int
    idx_to_uid = unique_users_sorted.to_numpy()

    # ---- Group bins and extract per-bin user index lists ----
    print("  > Grouping bins for pair generation...")
    bins_grouped = (
        user_bin_df.group_by(["tile_id", "window_id"])
        .agg(pl.col("_idx"))
    )
    user_idx_lists = bins_grouped["_idx"].to_list()
    n_bins = len(user_idx_lists)
    del user_bin_df, bins_grouped  # free memory

    # ---- Compute subsampling threshold from 16 GB memory limit ----
    # Avoids memory errors by subsampling bins with more than this max number of users
    # Peak memory per bin during pair generation: ~24 bytes per pair
    # (np.triu_indices returns 2 int64 arrays + 2 int32 indexed arrays)
    MAX_BIN_MEM_BYTES = 16 * 1024**3
    BYTES_PER_PAIR_PEAK = 24
    max_pairs_per_bin = MAX_BIN_MEM_BYTES // BYTES_PER_PAIR_PEAK
    max_bin_users = int(np.sqrt(2 * max_pairs_per_bin)) + 1
    print(f"  > Subsampling threshold: bins with > {max_bin_users:,} users "
          f"will be randomly subsampled")

    # ---- Generate pairwise co-occurrence edges, bin by bin ----
    # Building batches of pair observations and periodically flushing to the CSR matrix to keep memory usage bounded
    # Accumulates pair observations into a scipy sparse matrix in batches,
    # avoiding the O(N^2) intermediate explosion of the full self-join.
    print(f"  > Generating pairwise edges from {n_bins:,} bins...")
    BATCH_FLUSH_PAIRS = 50_000_000  # flush to CSR accumulator every 50M pairs
    accumulated_csr = None
    batch_rows = []
    batch_cols = []
    batch_count = 0
    total_pairs_generated = 0
    bins_subsampled = 0
    flush_count = 0
    rng = np.random.default_rng(seed=42)

    for i, idx_list in enumerate(user_idx_lists):
        n = len(idx_list)
        # Skip bins with fewer than 2 users (no pairs possible)
        if n < 2:
            continue

        idx = np.array(idx_list, dtype=np.int32)

        # Subsample if bin exceeds memory threshold
        if n > max_bin_users:
            # randomly subsample the bin to the max number of users
            idx = rng.choice(idx, size=max_bin_users, replace=False)
            idx.sort()
            n = max_bin_users
            bins_subsampled += 1

        # Generate upper-triangle pairs (i < j) for this bin
        ii, jj = np.triu_indices(n, k=1)
        # idx[ii] and idx[jj] are the user indices for the pairs
        batch_rows.append(idx[ii])
        batch_cols.append(idx[jj])
        n_pairs = len(ii)
        batch_count += n_pairs
        total_pairs_generated += n_pairs

        # Flush accumulated batch into CSR to keep memory bounded
        if batch_count >= BATCH_FLUSH_PAIRS:
            rows_arr = np.concatenate(batch_rows)
            cols_arr = np.concatenate(batch_cols)
            data_arr = np.ones(len(rows_arr), dtype=np.int32)
            # Convert to COO matrix and then to CSR matrix to keep memory bounded
            batch_coo = coo_matrix(
                (data_arr, (rows_arr, cols_arr)),
                shape=(n_graph_users, n_graph_users)
            )
            # Convert to CSR matrix to keep memory bounded
            # Compressed Sparse Row is more efficient for storage and arithmetic 
            batch_csr = batch_coo.tocsr()
            if accumulated_csr is None:
                accumulated_csr = batch_csr
            else:
                accumulated_csr += batch_csr
            # Free memory
            del rows_arr, cols_arr, data_arr, batch_coo, batch_csr
            # Reset batch
            batch_rows = []
            batch_cols = []
            batch_count = 0
            # Increment flush count
            flush_count += 1

        # Progress reporting so I know its not stuck
        if (i + 1) % 200_000 == 0:
            print(f"    ... processed {i+1:,}/{n_bins:,} bins "
                  f"({total_pairs_generated:,.0f} pair observations)")

    # Final flush of remaining batch
    if batch_rows:
        rows_arr = np.concatenate(batch_rows)
        cols_arr = np.concatenate(batch_cols)
        data_arr = np.ones(len(rows_arr), dtype=np.int32)
        batch_coo = coo_matrix(
            (data_arr, (rows_arr, cols_arr)),
            shape=(n_graph_users, n_graph_users)
        )
        batch_csr = batch_coo.tocsr()
        if accumulated_csr is None:
            accumulated_csr = batch_csr
        else:
            accumulated_csr += batch_csr
        del rows_arr, cols_arr, data_arr, batch_coo, batch_csr
        flush_count += 1

    print(f"  > {total_pairs_generated:,} total pair observations "
          f"({flush_count} batch flushes)")
    if bins_subsampled:
        print(f"  > {bins_subsampled:,} bins subsampled "
              f"(exceeded {max_bin_users:,} user threshold)")

    # ---- Extract edges above co-occurrence threshold ----
    # Returns numpy arrays instead of a Python dict to avoid large overhead
    edge_data = None
    if accumulated_csr is not None:
        final_coo = accumulated_csr.tocoo()
        n_unique_pairs = final_coo.nnz
        mask = final_coo.data >= MIN_CO_OCCURRENCE
        edge_src = final_coo.row[mask].astype(np.int32)
        edge_tgt = final_coo.col[mask].astype(np.int32)
        edge_wt = final_coo.data[mask].astype(np.int32)

        print(f"  > {n_unique_pairs:,} unique user pairs before thresholding")
        print(f"  > {len(edge_src):,} edges after threshold "
              f"(>= {MIN_CO_OCCURRENCE})")

        # Pack as tuple: (src_indices, tgt_indices, weights, index→uid map, vertex count)
        edge_data = (edge_src, edge_tgt, edge_wt, idx_to_uid, n_graph_users)
        del accumulated_csr, final_coo, mask
    else:
        print("  > No co-occurrence pairs generated")

    elapsed = time.perf_counter() - start
    print(f"  > Phase 3 complete: {elapsed:.1f}s")
    return edge_data, user_tiles


# ============================================================
# PHASE 4: Leiden Community Detection
# ============================================================
def phase4_leiden(edge_data, flagged_ids):
    """Run Leiden community detection on the co-occurrence graph.

    edge_data is a tuple (edge_src, edge_tgt, edge_wt, idx_to_uid, n_vertices)
    where edge_src/tgt are contiguous 0-based vertex indices from Phase 3's
    sparse matrix, idx_to_uid maps those indices back to user_id_int, and
    edge_wt holds the co-occurrence weights.

    To avoid creating ~33 GB of Python tuple objects for 298M edges, we write
    the edge list to a temp file and let igraph's C reader load it directly.
    """
    import tempfile
    import os

    print("\n" + "=" * 60)
    print("PHASE 4: Leiden Community Detection (CPM)")
    print("=" * 60)
    start = time.perf_counter()

    edge_src, edge_tgt, edge_wt, idx_to_uid, n_vertices = edge_data
    n_edges = len(edge_src)

    print(f"  > Graph: {n_vertices:,} vertices, {n_edges:,} edges")

    # ---- Write edge list to temp file ----
    # Polars write_csv is Rust-speed (~10-30s for 298M rows).
    # This avoids creating 298M Python tuples (~33 GB) in memory.
    print(f"  > Writing edge list to temp file...")
    temp_fd, temp_path = tempfile.mkstemp(suffix='.edgelist', prefix='botnet_')
    os.close(temp_fd)
    try:
        pl.DataFrame({
            "src": edge_src,
            "tgt": edge_tgt,
        }).write_csv(temp_path, separator=' ', include_header=False)

        # ---- Load graph via igraph's C reader ----
        print(f"  > Loading graph from edge list (C reader)...")
        g = ig.Graph.Read_Edgelist(temp_path, directed=False)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Ensure correct vertex count (isolated flagged users won't appear in edges)
    if g.vcount() < n_vertices:
        g.add_vertices(n_vertices - g.vcount())

    # ---- Assign edge weights ----
    # .tolist() creates ~8 GB of Python ints for 298M edges, but this is
    # temporary and freed after igraph copies them into its C structure.
    print(f"  > Assigning edge weights...")
    g.es["weight"] = edge_wt.tolist()

    # Free numpy arrays -- igraph owns the data now
    del edge_src, edge_tgt, edge_wt

    # ---- Run Leiden with CPM quality function ----
    print(f"  > Running Leiden algorithm (resolution={LEIDEN_RESOLUTION})...")
    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=LEIDEN_RESOLUTION,
    )

    # ---- Build community map ----
    # idx_to_uid maps graph vertex index → original user_id_int
    community_map = {}
    for vtx_idx, comm_id in enumerate(partition.membership):
        if vtx_idx < len(idx_to_uid):
            uid = int(idx_to_uid[vtx_idx])
            community_map[uid] = comm_id

    # Assign singleton communities to flagged users not in the graph
    max_comm_id = max(community_map.values()) if community_map else -1
    for uid in flagged_ids:
        if uid not in community_map:
            max_comm_id += 1
            community_map[uid] = max_comm_id

    community_df = pl.DataFrame({
        "user_id_int": list(community_map.keys()),
        "community_id": list(community_map.values()),
    })

    n_communities = community_df["community_id"].n_unique()
    sizes = community_df.group_by("community_id").agg(pl.len().alias("sz"))
    multi_member = sizes.filter(pl.col("sz") >= MIN_COMMUNITY_SIZE).height

    # Free the graph to reclaim memory before Phase 5
    del g

    elapsed = time.perf_counter() - start
    print(f"  > {n_communities:,} total communities, "
          f"{multi_member:,} with >= {MIN_COMMUNITY_SIZE} members")
    print(f"  > Modularity: {partition.modularity:.4f}")
    print(f"  > Phase 4 complete: {elapsed:.1f}s")
    return community_df


# ============================================================
# PHASE 5: Community Scoring
# ============================================================
def phase5_score_communities(community_df, user_features, user_tiles):
    """Score each community and flag likely botnets."""
    print("\n" + "=" * 60)
    print("PHASE 5: Community Scoring")
    print("=" * 60)
    start = time.perf_counter()

    # Ensure user_id_int types match for joins
    uid_dtype = user_features["user_id_int"].dtype
    community_df = community_df.cast({"user_id_int": uid_dtype})

    # Join community assignments with per-user features
    comm_features = community_df.join(user_features, on="user_id_int", how="left")

    # ---- Aggregate features per community ----
    print("  > Aggregating per-community statistics...")
    community_scores = (
        comm_features.group_by("community_id")
        .agg([
            pl.len().alias("community_size"),
            pl.col("pct_near_cooldown_5m").mean().alias("avg_pct_cooldown_5m"),
            pl.col("std_interval").mean().alias("avg_std_interval"),
            pl.col("std_interval").std().alias("std_of_std_interval"),
            pl.col("flag_count").mean().alias("flag_density"),
            pl.col("mean_placement_time").std().alias("temporal_coherence_std"),
            pl.col("mean_placement_time").mean().alias("avg_mean_time"),
            pl.col("total_placements").sum().alias("total_community_pixels"),
        ])
        .filter(pl.col("community_size") >= MIN_COMMUNITY_SIZE)
    )

    # Fill nulls in std columns (single-member or uniform communities)
    community_scores = community_scores.with_columns([
        pl.col("std_of_std_interval").fill_null(0.0).fill_nan(0.0),
        pl.col("temporal_coherence_std").fill_null(0.0).fill_nan(0.0),
    ])

    # ---- Spatial coherence (O(n) tile-frequency method) ----
    print("  > Computing spatial coherence per community...")
    comm_tiles = community_df.join(user_tiles, on="user_id_int", how="inner")

    if comm_tiles.height > 0 and "tile_id" in comm_tiles.columns:
        exploded = comm_tiles.explode("tile_id")
        tile_counts = (
            exploded.group_by(["community_id", "tile_id"])
            .agg(pl.len().alias("member_count"))
        )

        # Join community sizes
        comm_sizes = (
            community_df.group_by("community_id")
            .agg(pl.len().alias("_comm_size"))
        )
        tile_counts = tile_counts.join(comm_sizes, on="community_id", how="left")

        # Mark active tiles (>= COMMUNITY_TILE_PRESENCE_PCT of community present)
        tile_counts = tile_counts.with_columns(
            (pl.col("member_count")
             >= (pl.col("_comm_size").cast(pl.Float64)
                 * COMMUNITY_TILE_PRESENCE_PCT))
            .alias("is_active")
        )

        spatial_coherence = (
            tile_counts.group_by("community_id")
            .agg([
                pl.col("is_active").sum().alias("active_tiles"),
                pl.len().alias("total_tiles"),
                pl.col("tile_id")
                .sort_by("member_count", descending=True)
                .head(5).alias("top_tile_ids"),
                pl.col("member_count")
                .sort_by("member_count", descending=True)
                .head(5).alias("top_tile_counts"),
            ])
            .with_columns(
                (pl.col("active_tiles").cast(pl.Float64)
                 / pl.col("total_tiles").cast(pl.Float64))
                .alias("spatial_coherence")
            )
        )

        community_scores = community_scores.join(
            spatial_coherence.select([
                "community_id", "spatial_coherence",
                "top_tile_ids", "top_tile_counts",
                "active_tiles", "total_tiles",
            ]),
            on="community_id", how="left"
        )
    else:
        community_scores = community_scores.with_columns([
            pl.lit(0.0).alias("spatial_coherence"),
            pl.lit(None).cast(pl.List(pl.Int32)).alias("top_tile_ids"),
            pl.lit(None).cast(pl.List(pl.UInt32)).alias("top_tile_counts"),
            pl.lit(0).cast(pl.UInt32).alias("active_tiles"),
            pl.lit(0).cast(pl.UInt32).alias("total_tiles"),
        ])

    # ---- Flag botnet communities ----
    print("  > Evaluating botnet criteria...")
    community_scores = community_scores.with_columns(
        (
            # Criterion 1: high flag density
            (pl.col("flag_density") > BOTNET_FLAG_DENSITY_THRESHOLD)
            # Criterion 2: cooldown-synchronized group
            | (
                (pl.col("community_size") >= BOTNET_MIN_SIZE_FOR_COOLDOWN)
                & (pl.col("avg_pct_cooldown_5m") > BOTNET_COOLDOWN_THRESHOLD)
                & (pl.col("temporal_coherence_std") < BOTNET_TEMPORAL_STD_THRESHOLD)
            )
            # Criterion 3: behaviorally uniform large group
            | (
                (pl.col("community_size") >= BOTNET_MIN_SIZE_FOR_UNIFORMITY)
                & (pl.col("std_of_std_interval") < BOTNET_UNIFORMITY_STD_THRESHOLD)
            )
        ).alias("is_botnet")
    )

    n_botnets = community_scores.filter(pl.col("is_botnet")).height
    botnet_community_ids = set(
        community_scores.filter(pl.col("is_botnet"))["community_id"].to_list()
    )

    elapsed = time.perf_counter() - start
    print(f"  > {community_scores.height:,} communities scored, "
          f"{n_botnets:,} flagged as botnets")
    print(f"  > Phase 5 complete: {elapsed:.1f}s")
    return community_scores, botnet_community_ids


# ============================================================
# PHASE 6: Final Classification & Report
# ============================================================
def phase6_classify_and_report(user_features, community_df,
                               community_scores, botnet_community_ids,
                               total_unique_users):
    """Compute per-user bot scores and print the final report."""
    print("\n" + "=" * 60)
    print("PHASE 6: Final Classification & Report")
    print("=" * 60)

    # ---- Compute behavioral score ----
    # Only use actual boolean flag columns (not flag_count which is UInt8)
    flag_cols = [c for c in user_features.columns
                 if c.startswith("flag_") and c != "flag_count"]
    score_expr = pl.lit(0.0)
    for fc in flag_cols:
        weight = SCORE_WEIGHTS.get(fc, 0.0)
        if weight > 0:
            score_expr = score_expr + pl.col(fc).cast(pl.Float64) * weight

    user_features = user_features.with_columns(
        score_expr.alias("behavioral_score")
    )

    # ---- Add community score ----
    if community_df is not None and community_scores is not None:
        # Ensure dtype compatibility for joins
        uid_dtype = user_features["user_id_int"].dtype
        community_df = community_df.cast({"user_id_int": uid_dtype})

        # Get community size and botnet status per user
        comm_info = community_df.join(
            community_scores.select([
                "community_id", "community_size", "is_botnet"
            ]),
            on="community_id", how="left"
        )

        # Compute community score
        comm_info = comm_info.with_columns(
            pl.when(
                pl.col("is_botnet").fill_null(False)
                & (pl.col("community_size").fill_null(0) >= LARGE_COMMUNITY_SIZE)
            ).then(SCORE_LARGE_BOTNET_COMMUNITY)
            .when(pl.col("is_botnet").fill_null(False))
            .then(SCORE_BOTNET_COMMUNITY)
            .otherwise(0.0)
            .alias("community_score")
        )

        user_features = user_features.join(
            comm_info.select([
                "user_id_int", "community_id",
                "community_score", "is_botnet"
            ]),
            on="user_id_int", how="left"
        )
        user_features = user_features.with_columns([
            pl.col("community_score").fill_null(0.0),
            pl.col("is_botnet").fill_null(False),
        ])
    else:
        user_features = user_features.with_columns([
            pl.lit(0.0).alias("community_score"),
            pl.lit(None).cast(pl.Int64).alias("community_id"),
            pl.lit(False).alias("is_botnet"),
        ])

    # ---- Total bot score ----
    user_features = user_features.with_columns(
        (pl.col("behavioral_score") + pl.col("community_score"))
        .alias("bot_score")
    )

    # ---- Classification ----
    user_features = user_features.with_columns(
        pl.when(pl.col("bot_score") >= HIGH_CONFIDENCE_SCORE)
        .then(pl.lit("HIGH_CONFIDENCE"))
        .when(pl.col("bot_score") >= PROBABLE_BOT_SCORE)
        .then(pl.lit("PROBABLE"))
        .otherwise(pl.lit("UNLIKELY"))
        .alias("classification")
    )

    # ==================================================================
    # REPORT
    # ==================================================================
    total_analyzed = user_features.height
    flagged = user_features.filter(pl.col("flag_count") > 0).height
    high_conf = user_features.filter(
        pl.col("classification") == "HIGH_CONFIDENCE").height
    probable = user_features.filter(
        pl.col("classification") == "PROBABLE").height
    unlikely_flagged = flagged - high_conf - probable

    print("\n" + "=" * 60)
    print("  BOTNET ANALYSIS REPORT")
    print("=" * 60)
    print(f"\n  GLOBAL STATISTICS:")
    print(f"    Total unique accounts in dataset:  {total_unique_users:,}")
    print(f"    Accounts analyzed (>= {MIN_PLACEMENTS} placements): "
          f"{total_analyzed:,}")
    print(f"    Accounts flagged by behavior:      {flagged:,} "
          f"({flagged / total_unique_users * 100:.2f}% of all users)")

    if community_scores is not None:
        total_comms = community_scores.height
        n_botnets = community_scores.filter(pl.col("is_botnet")).height
        print(f"    Leiden communities (>= {MIN_COMMUNITY_SIZE} members): "
              f"{total_comms:,}")
        print(f"    Communities flagged as botnets:    {n_botnets:,}")

    print(f"\n  BOT CLASSIFICATION (as % of {total_unique_users:,} total users):")
    print(f"    High confidence (score >= {HIGH_CONFIDENCE_SCORE}): "
          f"{high_conf:,} ({high_conf / total_unique_users * 100:.2f}%)")
    print(f"    Probable bot    (score >= {PROBABLE_BOT_SCORE}): "
          f"{probable:,} ({probable / total_unique_users * 100:.2f}%)")
    print(f"    Unlikely bot    (flagged, score < {PROBABLE_BOT_SCORE}): "
          f"{unlikely_flagged:,}")
    print(f"    TOTAL LIKELY BOTS (high + probable): "
          f"{high_conf + probable:,} "
          f"({(high_conf + probable) / total_unique_users * 100:.2f}%)")

    print(f"\n  BEHAVIORAL FLAG BREAKDOWN:")
    for fc in flag_cols:
        count = user_features.filter(pl.col(fc).cast(pl.Boolean)).height
        print(f"    {fc:<28} {count:>10,} users "
              f"({count / total_unique_users * 100:.2f}%)")

    # ---- Top botnet communities ----
    if community_scores is not None and botnet_community_ids:
        top_botnets = (
            community_scores
            .filter(pl.col("is_botnet"))
            .sort("community_size", descending=True)
            .head(TOP_COMMUNITIES_TO_REPORT)
        )

        print(f"\n" + "-" * 60)
        print(f"  TOP {min(TOP_COMMUNITIES_TO_REPORT, top_botnets.height)} "
              f"BOTNET COMMUNITIES")
        print("-" * 60)

        for i, row in enumerate(top_botnets.iter_rows(named=True), 1):
            comm_id = row["community_id"]
            size = row["community_size"]
            flag_d = row["flag_density"]
            avg_cd5 = row["avg_pct_cooldown_5m"]
            avg_std = row["avg_std_interval"]
            std_std = row["std_of_std_interval"]
            tc_std = row["temporal_coherence_std"]
            sc = row.get("spatial_coherence", 0.0) or 0.0
            active_t = row.get("active_tiles", 0) or 0
            total_t = row.get("total_tiles", 0) or 0
            total_px = row["total_community_pixels"]

            # Convert avg_mean_time to event hours
            avg_time_hr = (row["avg_mean_time"] or 0) / 3600.0

            print(f"\n  {i}. Community #{comm_id} "
                  f"({size:,} members, {total_px:,} pixels)")
            print(f"     Flag density:           {flag_d:.2f}")
            print(f"     Avg cooldown precision: 5m={avg_cd5:.2f}")
            print(f"     Avg std interval:       {avg_std:.1f}s "
                  f"(uniformity std: {std_std:.1f}s)")
            print(f"     Temporal coherence:     "
                  f"{tc_std:.0f}s std (center: hour {avg_time_hr:.1f})")
            print(f"     Spatial coherence:      "
                  f"{sc:.2f} ({active_t}/{total_t} active tiles)")

            # Top 5 target tiles
            top_ids = row.get("top_tile_ids")
            top_counts = row.get("top_tile_counts")
            if top_ids and top_counts:
                print(f"     Top target tiles:")
                for j, (tid, tc) in enumerate(
                        zip(top_ids[:5], top_counts[:5]), 1):
                    x0, x1, y0, y1 = tile_id_to_ranges(tid)
                    print(f"       {j}. x: {x0}-{x1}, y: {y0}-{y1} "
                          f"({tc} members)")

    print("\n" + "=" * 60)
    print("  END OF REPORT")
    print("=" * 60)

    return user_features


# ============================================================
# MAIN
# ============================================================
def main():
    overall_start = time.perf_counter()

    # Phase 1: Extract per-user features
    user_features, total_unique_users = phase1_extract_features(DATA_FILE_PATH)

    # Phase 2: Behavioral flagging
    user_features, flagged_ids = phase2_flag_users(user_features)

    if not flagged_ids:
        print("\nNo users flagged by behavioral signals. Analysis complete.")
        return

    # Phase 3: Build co-occurrence graph
    edge_data, user_tiles = phase3_build_graph(DATA_FILE_PATH, flagged_ids)

    if edge_data is None:
        print("\nNo co-occurrence edges found above threshold. "
              "Reporting behavioral flags only.")
        phase6_classify_and_report(
            user_features, None, None, set(), total_unique_users
        )
        return

    # Phase 4: Leiden community detection
    community_df = phase4_leiden(edge_data, flagged_ids)
    del edge_data  # free numpy arrays after graph is built

    # Phase 5: Score communities
    community_scores, botnet_community_ids = phase5_score_communities(
        community_df, user_features, user_tiles
    )

    # Phase 6: Final classification and report
    phase6_classify_and_report(
        user_features, community_df, community_scores,
        botnet_community_ids, total_unique_users
    )

    # Phase 7: Generate botnet visualizations
    if community_scores is not None and botnet_community_ids:
        generate_botnet_plots(
            DATA_FILE_PATH, community_df, community_scores,
            n_top=TOP_COMMUNITIES_TO_REPORT
        )
        plot_all_botnets(
            DATA_FILE_PATH, community_df, community_scores
        )

    elapsed = time.perf_counter() - overall_start
    print(f"\nTotal analysis time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
