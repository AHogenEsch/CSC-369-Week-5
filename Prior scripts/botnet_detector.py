import sys
import polars as pl
import numpy as np
import time
from scipy.ndimage import label, generate_binary_structure

DATA_FILE_PATH = 'processed_place_data.parquet'
RPLACE_WIDTH = 2000
RPLACE_HEIGHT = 2000

# CONFIGURATION
# Percentile for defining a "Hot Zone" on the canvas (Higher = stricter/smaller zones)
ZONE_PERCENTILE = 0.97 
# Minimum pixels placed in ONE SECOND in ONE ZONE to trigger a detection
DENSITY_THRESHOLD = 50
# If an attack pauses for this many seconds, the next wave is considered a NEW event
ATTACK_GAP_SECONDS = 150 

def build_dynamic_zones(lf):
    print("  > Building Activity Heatmap...")
    heatmap = (
        lf.group_by(["x", "y"])
        .agg(pl.len().alias("pixel_count"))
        .collect()
    )
    
    threshold = heatmap["pixel_count"].quantile(ZONE_PERCENTILE)
    print(f"  > Zone Threshold: {threshold:.0f} pixels ({ZONE_PERCENTILE*100}th percentile)")
    
    canvas_matrix = np.zeros((RPLACE_WIDTH, RPLACE_HEIGHT), dtype=int)
    in_bounds_filter = (pl.col("x") < RPLACE_WIDTH) & (pl.col("y") < RPLACE_HEIGHT)
    
    hot_pixels = heatmap.filter((pl.col("pixel_count") > threshold) & in_bounds_filter)
    xs = hot_pixels["x"].to_numpy()
    ys = hot_pixels["y"].to_numpy()
    canvas_matrix[xs, ys] = 1 
    
    print("  > Clustering Hot Zones...")
    structure = generate_binary_structure(2, 2) 
    labeled_array, num_features = label(canvas_matrix, structure=structure)
    print(f"  > Identified {num_features} distinct high-traffic zones.")
    
    active_xs, active_ys = np.where(labeled_array > 0)
    zone_ids = labeled_array[active_xs, active_ys]
    
    zone_lookup = pl.DataFrame({
        "x": active_xs,
        "y": active_ys,
        "zone_id": zone_ids
    }).with_columns([
        pl.col("x").cast(pl.Int16),
        pl.col("y").cast(pl.Int16),
        pl.col("zone_id").cast(pl.Int32)
    ])
    
    return zone_lookup

def detect_coordinated_botnets(file_path=DATA_FILE_PATH):
    start_perf = time.perf_counter_ns()
    
    try:
        lf = pl.scan_parquet(file_path)
        
        # 1. Build Map
        zone_map = build_dynamic_zones(lf)
        lf_zones = zone_map.lazy()
        
        print(f"  > Scanning for synchronized bursts (Density > {DENSITY_THRESHOLD} px/sec)...")

        # 2. Localized Spike Detection
        # Instead of finding GLOBAL spikes, we calculate density PER ZONE PER SECOND.
        # This prevents flagging random users when the global board is busy.
        zone_seconds = (
            lf.join(lf_zones, on=["x", "y"], how="inner")
            .group_by(["zone_id", "seconds_since_start"])
            .agg([
                pl.len().alias("zone_density"),
                pl.col("user_id_int") # Capture the specific users in this second
            ])
            # STRICT FILTER: Only keep seconds where a single zone saw massive activity
            .filter(pl.col("zone_density") > DENSITY_THRESHOLD)
        )
        
        # Performance Note: The result of this filter is small (only attack seconds).
        # We collect here to perform the complex "Windowing/Gap Detection" in eager mode,
        # which is much faster/stable than trying to optimize it on the Lazy graph.
        attacks_df = zone_seconds.collect()
        
        if attacks_df.height == 0:
            print("No botnet bursts found with current thresholds.")
            return

        print(f"  > Found {attacks_df.height} high-density seconds. Grouping into coordinated events...")

        # 3. Burst Clustering (Sessionization)
        # We sort by Zone and Time, then look for gaps to identify distinct "Events"
        coordinated_events = (
            attacks_df.sort(["zone_id", "seconds_since_start"])
            .with_columns([
                # Calculate time gap between current attack-second and previous one IN THE SAME ZONE
                (pl.col("seconds_since_start") - pl.col("seconds_since_start").shift(1))
                .over("zone_id").fill_null(9999).alias("time_diff")
            ])
            .with_columns([
                # If gap > 60s, increment the "Burst ID"
                (pl.col("time_diff") > ATTACK_GAP_SECONDS).cum_sum().over("zone_id").alias("burst_id")
            ])
            # Create a unique ID for this specific attack event (ZoneID_BurstID)
            .with_columns(
                pl.format("{}_{}", pl.col("zone_id"), pl.col("burst_id")).alias("unique_event_id")
            )
        )

        # 4. Analysis & Reporting
        # We aggregate based on these new Unique Events
        top_events = (
            coordinated_events.group_by("unique_event_id")
            .agg([
                pl.col("zone_id").first().alias("zone_id"),
                pl.sum("zone_density").alias("total_pixels"),
                # Count UNIQUE users involved in this specific burst
                pl.col("user_id_int").explode().unique().len().alias("unique_bots"),
                pl.min("seconds_since_start").alias("start_sec"),
                pl.max("seconds_since_start").alias("end_sec"),
                pl.len().alias("active_seconds")
            ])
            .sort("total_pixels", descending=True)
            .limit(10) # Show Top distinct attacks
        )

        # Calculate Total Unique Users across ALL detected bursts
        total_unique_bots = (
            coordinated_events.select(pl.col("user_id_int").explode().unique())
        ).height

        exec_time = (time.perf_counter_ns() - start_perf) // 1_000_000
        
        print(f"\n" + "="*60)
        print(f"COORDINATED BOTNET REPORT")
        print(f"="*60)
        print(f"Global Stats:")
        print(f"  Unique Suspicious Users: {total_unique_bots:,}")
        print(f"  Distinct Attack Bursts:  {coordinated_events['unique_event_id'].n_unique():,}")
        print(f"  Processing Time:         {exec_time} ms")
        print(f"-"*60)
        print(f"TOP 10 COORDINATED ATTACK EVENTS:")
        
        for i, row in enumerate(top_events.rows(named=True), 1):
            z_id = row['zone_id']
            u_bots = row['unique_bots']
            pixels = row['total_pixels']
            duration = row['end_sec'] - row['start_sec']
            
            # Get zone coordinates for context
            z_pixels = zone_map.filter(pl.col("zone_id") == z_id)
            min_x, max_x = z_pixels["x"].min(), z_pixels["x"].max()
            min_y, max_y = z_pixels["y"].min(), z_pixels["y"].max()
            center_x = int((min_x + max_x) / 2)
            center_y = int((min_y + max_y) / 2)
            
            # Readable Time
            start_hr = row['start_sec'] / 3600
            end_hr = row['end_sec'] / 3600

            print(f"\n{i}. ATTACK ON ZONE {z_id} (Burst ID: {row['unique_event_id']})")
            print(f"   Target Location: ({center_x}, {center_y}) [Bounds: {min_x}-{max_x}, {min_y}-{max_y}]")
            print(f"   Coordinated Size: {u_bots:,} unique users")
            print(f"   Volume:           {pixels:,} pixels")
            print(f"   Duration:         {duration} seconds, or {duration/60:.2f} minutes, (Hours {start_hr:.2f} - {end_hr:.2f})")
            
        print(f"="*60)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    detect_coordinated_botnets()