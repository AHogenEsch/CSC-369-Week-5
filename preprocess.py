import polars as pl
import time
from datetime import datetime

# Configuration
DATA_FILE_PATH = '2022_place_canvas_history.csv'
OUTPUT_FILE_PATH = 'processed_place_data.parquet'
RPLACE_WIDTH = 2000
EVENT_START = '2022-04-01 12:44:10 UTC'

# The 32-color palette mapping for 2022
COLOR_MAP = {
    "#6D001A": "dark red", "#BE0039": "red", "#FF4500": "orange", "#FFA800": "yellow",
    "#FFD635": "pale yellow", "#FFF8B8": "ivory", "#00A368": "dark green", "#00CC78": "green",
    "#7EED56": "light green", "#00756F": "dark teal", "#009EAA": "teal", "#00CCC0": "light teal",
    "#2450A4": "dark blue", "#3690EA": "blue", "#51E9F4": "light blue", "#493AC1": "indigo",
    "#6A5CFF": "periwinkle", "#94B3FF": "lavender", "#811E9F": "dark purple", "#B44AC0": "purple",
    "#E4ABFF": "pale purple", "#DE107F": "magenta", "#FF3881": "pink", "#FF99AA": "light pink",
    "#6D482F": "dark brown", "#9C6926": "brown", "#FFB470": "beige", "#000000": "black",
    "#515252": "dark gray", "#898D90": "gray", "#D4D7D9": "light gray", "#FFFFFF": "white"
}

def preprocess():
    print(f"Starting Preprocessing: {DATA_FILE_PATH}...")
    start_perf = time.perf_counter_ns()

    # Convert Event Start
    try:
        event_start_dt = datetime.strptime(EVENT_START, "%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        print("Error parsing EVENT_START. Ensure format matches '2022-04-01 12:44:10 UTC'")
        return

    # Build Lazy Computation Graph
    q = pl.scan_csv(DATA_FILE_PATH)

    # --- Timestamps & User IDs ---
    q = q.with_columns([
        # Calculate seconds since start, we need to calculate average session time in seconds. Smaller value than ms after epoc
        ((pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.f %Z") - event_start_dt)
         .dt.total_seconds()).cast(pl.Int32).alias("seconds_since_start"),

        # Map User IDs to integers
        pl.col("user_id").rank("dense").alias("user_id_int")
    ])

    # --- Colors ---
    # Use color map to reduce cardinality
    q = q.with_columns(
        pl.col("pixel_color").replace(COLOR_MAP).fill_null("unknown").alias("color_name")
    )

    # --- Coordinates --- REVISED WEEK 4
    # Add a temporary column to identify the coordinate count (2 for normal operation, 4 for moderator actions)
    q = q.with_columns(
        pl.col("coordinate").str.split(",").alias("coords_list")
    ).with_columns(
        pl.col("coords_list").list.len().alias("c_count")
    )

    # Process Standard Pixels (normal users)
    standard_pixels = q.filter(pl.col("c_count") == 2).with_columns([
        pl.col("coords_list").list.get(0).cast(pl.Int16).alias("x"),
        pl.col("coords_list").list.get(1).cast(pl.Int16).alias("y")
    ])

    # Process Moderator Rectangles to create individual pixel placements
    # This creates 102,275 additional rows from 19 moderator rectangles.
    mod_pixels = q.filter(pl.col("c_count") == 4).with_columns([
        pl.col("coords_list").list.get(0).cast(pl.Int32).alias("x1"),
        pl.col("coords_list").list.get(1).cast(pl.Int32).alias("y1"),
        pl.col("coords_list").list.get(2).cast(pl.Int32).alias("x2"),
        pl.col("coords_list").list.get(3).cast(pl.Int32).alias("y2")
    ]).with_columns([
        pl.int_ranges(pl.col("x1"), pl.col("x2") + 1).alias("x_range"),
        pl.int_ranges(pl.col("y1"), pl.col("y2") + 1).alias("y_range")
    ]).explode("x_range").explode("y_range").select([
        pl.all().exclude(["x1", "y1", "x2", "y2", "x_range", "y_range"]),
        pl.col("x_range").cast(pl.Int16).alias("x"),
        pl.col("y_range").cast(pl.Int16).alias("y")
    ])

    # Merge: Vertical concatenation back into one stream
    q = pl.concat([standard_pixels, mod_pixels]).drop(["coords_list", "c_count"])

    # --- Final Selection & Optimization ---
    q = q.select([
        "seconds_since_start",
        "user_id_int",
        "color_name",
        "x",
        "y"
    ])

    # Sorting here allows the analysis script to skip the expensive sort operation.
    # Sort by User ID (grouping activity) then Time (ordering activity).
    # This enables linear scanning for sessionization and streaming for aggregations.
    q = q.sort(["user_id_int", "seconds_since_start"])

    # Execute and Save to Parquet
    print("Executing transformations and writing to Parquet...")
    
    # We use streaming=True to keep memory usage low, though the sort above 
    # will force Polars to manage a large buffer or spill to disk if memory is tight.
    q.sink_parquet(OUTPUT_FILE_PATH)

    exec_time = (time.perf_counter_ns() - start_perf) // 1_000_000_000
    print(f"Done! Preprocessed file saved to: {OUTPUT_FILE_PATH}")
    print(f"Processing time: {exec_time} seconds")

if __name__ == "__main__":
    preprocess()