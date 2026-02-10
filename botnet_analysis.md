# Botnet Analysis of r/place 2022: Technical Report

## Overview

In this analysis, I identify the percentage of users who are likely to be bots, as well as the number of coordinated botnets, with a much higher degree of confidence than last weeks' analysis. In order to identify users as bots with a high degree of confidence, users are first flagged for suspicious behaviour, then put into spatial and temporal bins. Community analysis can then be performed counting the number of co-occurances of users within the same tiles and timeframes, which is indicative that they are being used together to place pixels to create or destroy artwork.

Results:
> GLOBAL STATISTICS:
    Total unique accounts in dataset:  10,381,163
    Accounts analyzed (>= 10 placements): 3,807,345
    Accounts flagged by behavior:      216,625 (2.09% of all users)
    Leiden communities (>= 3 members): 1,377
    Communities flagged as botnets:    315

  BOT CLASSIFICATION (as % of 10,381,163 total users):
    High confidence (score >= 5.0): 3,755 (0.04%)
    Probable bot    (score >= 3.0): 8,917 (0.09%)
    Unlikely bot    (flagged, score < 3.0): 203,953
    TOTAL LIKELY BOTS (high + probable): 12,672 (0.12%)

  BEHAVIORAL FLAG BREAKDOWN:
    flag_cooldown                    93,891 users (0.90%)
    flag_low_interval_std             8,524 users (0.08%)
    flag_24h_active                   1,736 users (0.02%)
    flag_small_area                     568 users (0.01%)
    flag_single_color                15,198 users (0.15%)
    flag_printer                      8,689 users (0.08%)
    flag_high_volume                113,101 users (1.09%)

From an image of all the pixels from all botnets, it is clear that botnets were used throughout the entire canvas, specifically on the french and american flags at the bottom of the canvas, with large botnet footprints also on the trans flag and an artwork at (1500,600). I created PNGs depicting the five largest botnets as well as the total botnet footprint, which are included in the repo.


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
| `flag_printer` | `pct_adjacent > 0.7` and `pct_single_axis > 0.6` | Raster-scan movement pattern |
| `flag_high_volume` | `total_placements > P97` | Top 3% by volume — unusual activity level |

All users entering Phase 2 already have ≥ 10 placements (filtered in Phase 1), so no per-flag minimum placement thresholds are needed. A `flag_count` column sums the flags, and any user with `flag_count > 0` is considered "flagged" and proceeds to graph construction.

---

## Phase 3: Co-occurrence Graph Construction

This is the most computationally demanding phase and underwent the most significant redesigns during development.

### Concept

The canvas is divided into spatial-temporal bins: 50×50 pixel tiles × 5-minute time windows. If two flagged users both placed pixels in the same tile during the same time window, they co-occurred in that bin. The number of bins two users share becomes the edge weight in a co-occurrence graph. Edges below `MIN_CO_OCCURRENCE` (3) are discarded — if two users only overlapped in 1–2 bins, the co-occurrence is likely coincidental.

### Implementation