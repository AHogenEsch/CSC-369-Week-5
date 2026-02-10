# Botnet Analysis of r/place 2022: Technical Report

## Overview

In this analysis, I identify the percentage of users who are likely to be bots, as well as the number of coordinated botnets, with a much higher degree of confidence than last weeks' analysis. In order to identify users as bots with a high degree of confidence, users are first flagged for suspicious behaviour, then put into spatial and temporal bins. Community analysis can then be performed counting the number of co-occurances of users within the same tiles and timeframes, which is indicative that they are being used together to place pixels to create or destroy artwork. There are 6 distinct phases of this analysis, detailed below.

## Results:
> GLOBAL STATISTICS:
    Total unique accounts in dataset:  10,381,163
    Accounts analyzed (>= 10 placements): 3,807,345
    Accounts flagged by behavior:      216,625 (2.09% of all users)
    Leiden communities (>= 3 members): 1,388
    Communities flagged as botnets:    **314**

  BOT CLASSIFICATION (as % of 10,381,163 total users):
    High confidence (score >= 5.0): **11,901** (0.11%)
    Probable bot    (score >= 3.0): **85,676** (0.83%)
    Unlikely bot    (flagged, score < 3.0): 119,048
    TOTAL LIKELY BOTS (high + probable): **97,577** (0.94%)

  BEHAVIORAL FLAG BREAKDOWN:
    flag_cooldown                    93,891 users (0.90%)
    flag_low_interval_std             8,524 users (0.08%)
    flag_24h_active                   1,736 users (0.02%)
    flag_small_area                     568 users (0.01%)
    flag_single_color                15,198 users (0.15%)
    flag_printer                      8,689 users (0.08%)
    flag_high_volume                113,101 users (1.09%)

From an image of all the pixels from all botnets, it is clear that botnets were used throughout the entire canvas, specifically on the french and american flags at the bottom of the canvas, with large botnet footprints also on the trans flag and an artwork at (1500,600). About 1% of all r/place 2022 users were bots, a much more accurate figure than the 16.7% result that I found last week. I created PNGs depicting the five largest botnets as well as the total botnet footprint, which are included in the repo.


## Phase 1: User Feature Extraction

Phase 1 reads the Parquet file with Polars and computes behavioral features for every user with at least 10 pixel placements. Users below this threshold are excluded early, as they did not place enough pixels to be worth analyzing. This initial computation phase sets up values used in later stages of analysis.

All core features (timing, spatial, color, printer) are computed in a single `group_by("user_id_int").agg(...)` call. This ensures Polars reads the Parquet file once, applies all aggregations in parallel, and produces the user feature table in one materialization step. Features that cannot be expressed in a single aggregation (dominant color percentage, max continuous hours) are computed in separate targeted passes and joined back.

### Features Computed

**Timing features** (computed from the sorted diff of `seconds_since_start`):
- `total_placements`: raw count of pixels placed.
- `active_span_s`: time between first and last placement.
- `median_interval` and `std_interval`: central tendency and spread of inter-placement gaps. Bots tend to have very low standard deviation because they fire on a fixed timer.
- `pct_near_cooldown_5m`: fraction of intervals falling in the 295–310 second range (near the 5m cooldown). A high value indicates the account places a pixel as soon as the cooldown expires.

**Spatial features**:
- `x_std`, `y_std`: standard deviation of placement coordinates.
- `bounding_box_area`: `(x_max - x_min + 1) * (y_max - y_min + 1)`. A tiny bounding box (e.g., 9 = 3×3) combined with many placements suggests a bot repeatedly overwriting the same small region.
- `unique_pixels`: number of distinct (x, y) coordinates used.

**Color features**:
- `unique_colors`: number of distinct colors placed.
- `dominant_color_pct`: fraction of placements using the most common color. Computed via a separate double `group_by` (first by user+color to count, then by user to find the max), since this statistic cannot be expressed in a single aggregation pass.

**Printer detection features** (designed to catch bots that fill regions in printer-like patterns):
- `pct_adjacent`: fraction of consecutive placement pairs within Manhattan distance ≤ 2. Printer bots place pixels next to each other sequentially.
- `pct_single_axis_movement`: fraction of consecutive movements where either Δx = 0 or Δy = 0. Printer bots sweep along one axis before stepping to the next row/column.
- `sweep_score`: the average run length of same-direction horizontal or vertical movement. A high score indicates long unbroken sweeps. The dominant axis (max of x and y sweep scores) is kept. This metric was specifically designed to catch multi-line printer bots that a simpler "single straight line" check would miss.

**Continuous activity**:
- `max_continuous_hours`: the longest streak of consecutive hours with at least one placement. This uses a difference-from-sequence trick: for consecutive `hour_id` values like 5, 6, 7, subtracting the cumulative count 1, 2, 3 yields a constant (4). A gap in hours produces a new constant, naturally segmenting streaks. This approach is much more efficient than a complex sessionalization query.

---

## Phase 2: Behavioral Flagging

Phase 2 applies configurable thresholds to the feature table and assigns boolean flags:

| Flag | Condition | Rationale |
|------|-----------|-----------|
| `flag_cooldown` | `pct_near_cooldown_5m > 0.5` | >50% of intervals near the exact cooldown — mechanical timing |
| `flag_low_interval_std` | `std_interval < 15s` and `> 0` | Near-zero variance in placement timing — timer-driven |
| `flag_24h_active` | `max_continuous_hours > 24` | No human sustains hourly activity for 24+ hours without sleep |
| `flag_small_area` | `bounding_box_area ≤ 9` | Operating in a 3×3 pixel region — single-target bot |
| `flag_single_color` | `unique_colors == 1` | Only ever places one color — automated single-purpose account |
| `flag_printer` | `pct_adjacent > 0.7` and `pct_single_axis > 0.6` | printer movement pattern |
| `flag_high_volume` | `total_placements > P97` | Top 3% by volume — high activity level |

All users entering Phase 2 already have ≥ 10 placements (filtered in Phase 1). A `flag_count` column sums the flags, and any user with `flag_count > 0` is considered "flagged" and proceeds to graph construction.

---

## Phase 3: Co-occurrence Graph Construction

This is the most computationally demanding phase and underwent the most significant redesigns during development.

### Concept

The canvas is divided into spatial-temporal bins: 50×50 pixel tiles × 5-minute time windows. If two flagged users both placed pixels in the same tile during the same time window, they co-occurred in that bin. The number of bins two users share becomes the edge weight in a co-occurrence graph. Edges below `MIN_CO_OCCURRENCE` (3) are discarded — if two users only overlapped in 1–2 bins, the co-occurrence is likely coincidental.

### Implementation

1. **Filter to flagged users**: The Parquet is re-scanned, filtered via a semi-join to only flagged user IDs, and `tile_id` and `window_id` columns are created.

2. **Deduplicate user-bin assignments**: A user appearing multiple times in the same bin (possible due to cooldown edge cases) is counted only once via `.unique()`.

3. **Filter to valid bins**: Bins with fewer than 2 users are removed — no pairs can be generated from them.

4. **Map user IDs to contiguous indices**: The sparse matrix requires 0-based contiguous indices. A mapping DataFrame is joined onto the user-bin table, and a reverse lookup array (`idx_to_uid`) is stored for converting back to original user IDs later. 

5. **Group bins into per-bin user lists**: A `group_by(["tile_id", "window_id"]).agg(pl.col("_idx"))` produces a list-of-lists structure where each inner list contains the contiguous indices of users present in that bin.

6. **Iterate bin-by-bin, generating pairs with numpy**: For each bin, `np.triu_indices(n, k=1)` generates all upper-triangle index pairs (ensuring each pair is counted once with consistent ordering). These pairs are appended to batch buffers.

7. **Flush batches into a scipy CSR sparse matrix**: Every 50 million pairs, the batch is converted to a COO (Coordinate) sparse matrix, then to CSR (Compressed Sparse Row), and added to a running accumulator. COO-to-CSR conversion automatically sums duplicate entries — if users 2 and 7 appear together in 5 bins across the batch, the 5 separate (1, (2, 7)) entries become a single entry with value 5.

8. **Extract thresholded edges**: After all bins are processed, the accumulated CSR matrix is converted to COO, and entries with weight ≥ `MIN_CO_OCCURRENCE` are extracted as numpy arrays.

### Memory Optimization Design History

This phase underwent three major rewrites to minimize memory usage:

**Version 1 — Python Counter with itertools.combinations**: The original implementation grouped users per bin, generated all pairs using `itertools.combinations`, and counted co-occurrences in a `collections.Counter` dictionary. This crashed with a `MemoryError` because each Python tuple key `(u1, u2)` consumes ~150 bytes of object overhead. With hundreds of millions of unique pairs, the dictionary alone was huge.

**Version 2 — Polars self-join**: The Counter was replaced with a Polars self-join: `user_bin_df.join(user_bin_df, on=["tile_id", "window_id"]).filter(col("user_id_int") < col("user_id_int_r")).group_by(...).agg(...)`. This leverages Polars' Rust-based columnar engine, which is far more memory-efficient than Python dicts. However, the self-join still materializes all pairs before filtering — with ~31M user-bin rows and an average of 34 users per bin, the intermediate result was near 2 billion rows, again crashing.

**Version 3 (current) — Per-bin numpy iteration with scipy sparse accumulation**: The self-join was replaced with a Python loop that processes each bin individually. For each bin, `np.triu_indices` generates pairs entirely in numpy (C-level memory, ~24 bytes per pair vs. ~150 bytes for Python tuples). Pairs are accumulated into batch buffers and periodically flushed into a scipy sparse CSR matrix, which stores only non-zero entries (~12 bytes per unique pair). This approach keeps peak memory significantly lower, and a safety threshold subsamples any bin with more than ~37,000 users (where even one bin's pairs would exceed 16 GB), to ensure this would be the final draft of this phase.

---

## Phase 4: Leiden Community Detection

Phase 4 takes the graph data (about 216k vertices and 181 million edges) from Phase 3 and runs the Leiden algorithm to discover communities of co-occurring users.

### Algorithm Choice: CPM (Constant Potts Model)

The Leiden algorithm with CPM was chosen over modularity-based approaches for a specific reason: CPM does not suffer from the resolution limit. Modularity-based methods cannot detect communities smaller than a scale determined by the total graph size — they tend to merge small genuine communities into larger ones. Since botnets range from 3-member small-scale operations to 100+ member campaigns, CPM's ability to detect communities of varying sizes without bias was critical.

The CPM quality function is: `Q = Σ(internal_edges - resolution × C(n, 2))` for each community of size n. The resolution parameter (set to 0.02 for the final analysis) controls the minimum internal edge density a community must have to be considered valid. Lower values allow sparser communities; higher values require tighter coordination. I tested the analysis with several r values and found 0.02 to be a good resoultion that built sufficiently large communities.

### Memory Optimization

Phase 3 returns the edge data as three compact numpy arrays (`edge_src`, `edge_tgt`, `edge_wt`) along with the index-to-user-ID mapping. The original design returned a Python dictionary mapping `(u1, u2) → weight`, where each entry carries significant object overhead from the tuple key and the dict's hash table. Switching to numpy arrays reduced the memory footprint to roughly a tenth of the dictionary-based approach, since numpy stores values as contiguous raw integers with no per-element Python object overhead.

Even with numpy arrays, passing hundreds of millions of edges to igraph's `Graph()` constructor would require creating an equal number of Python tuple objects, because igraph's Python binding iterates edges as Python sequences. To avoid this, Phase 4 writes the edge list to a temporary file using Polars' fast `write_csv`, then loads it via `igraph.Graph.Read_Edgelist()`, which reads the file directly in C with zero Python object overhead. The temporary file is deleted after loading.

---

## Phase 5: Community Scoring

Phase 5 evaluates each community (with ≥ 3 members) and determines whether it exhibits botnet characteristics.

### Per-Community Metrics

For each community, the per-user features from Phase 1 are aggregated:
- **flag_density**: average number of behavioral flags per member. A community of users who each trigger 2–3 flags is far more suspicious than one where members have 0–1 flags.
- **avg_pct_cooldown_5m**: how precisely the community's members time their placements to the cooldown.
- **avg_std_interval / std_of_std_interval**: the average timing regularity and, critically, how *uniform* that regularity is across members. If all members have nearly identical `std_interval`, they are likely running the same automation script.
- **temporal_coherence_std**: standard deviation of members' mean placement times. A low value means the community's members were active during the same time period — they aren't just spatially co-located, they're temporally synchronized.
- **spatial_coherence**: computed via an O(N) tile-frequency method. For each tile the community touched, the fraction of members present is calculated. Tiles where ≥ 25% of the community was active are "active tiles." Spatial coherence = active_tiles / total_tiles. A high value means the community is focused on specific canvas regions rather than scattered randomly.

### Botnet Classification Criteria

A community is flagged as a botnet if it meets any of three criteria:
1. **High flag density** (> 1.5 average flags per member)
2. **Cooldown-synchronized group**: ≥ 5 members, average cooldown precision > 0.4, and temporal coherence std < 2 hours
3. **Behaviorally uniform large group**: ≥ 10 members with std_of_std_interval < 10 seconds (all members behave identically)

---

## Phase 6: Final Classification and Reporting

Phase 6 computes a composite bot score for every user:

**Behavioral score** = weighted sum of individual flags. Weights reflect confidence in each signal — `flag_cooldown` (3.0) and `flag_low_interval_std` (3.0) are weighted the highest because they are the best indicators of bot activity, inhumane timing precision over many placements. `flag_24h_active` (2.0) and `flag_printer` (2.0) are also weighted heavier because they are signs of automation. `flag_small_area` (1.0),`flag_single_color` (0.5), and `flag_high_volume` (0.5) are weighted lowest because they have higher false-positive rates.

**Community score** = 3.0 if the user belongs to a botnet community, or 4.0 if that community has ≥ 20 members (large botnets are higher confidence).

**Total bot score** = behavioral + community. Users are classified as:
- **HIGH_CONFIDENCE** (score ≥ 5.0): strong individual signals AND/OR botnet community membership
- **PROBABLE** (score ≥ 3.0): moderate signals, likely automated
- **LESS LIKELY** (score < 3.0): flagged but insufficient evidence, could still be bots but will contain more false positives

All percentages are reported against the total unique users in the dataset (~10M), not just the filtered subset, to get the overall bot percentage of all r/place activity.

---
