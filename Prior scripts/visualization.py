import polars as pl
import pyvista as pv
import numpy as np
import sys
import time
from datetime import datetime

# --- CONFIGURATION ---
DATA_FILE_PATH = '2022_place_canvas_history.csv' # Raw CSV path
RPLACE_WIDTH = 2000
RPLACE_HEIGHT = 2000
DEFAULT_TIME = '2022-04-04 23:59:59' # Default to end of event

# Color Palette Mapping (Hex -> RGB Tuple 0-255)
# This converts the hex strings in your CSV to usable texture colors
HEX_TO_RGB = {
    "#6D001A": (109, 0, 26), "#BE0039": (190, 0, 57), "#FF4500": (255, 69, 0), "#FFA800": (255, 168, 0),
    "#FFD635": (255, 214, 53), "#FFF8B8": (255, 248, 184), "#00A368": (0, 163, 104), "#00CC78": (0, 204, 120),
    "#7EED56": (126, 237, 86), "#00756F": (0, 117, 111), "#009EAA": (0, 158, 170), "#00CCC0": (0, 204, 192),
    "#2450A4": (36, 80, 164), "#3690EA": (54, 144, 234), "#51E9F4": (81, 233, 244), "#493AC1": (73, 58, 193),
    "#6A5CFF": (106, 92, 255), "#94B3FF": (148, 179, 255), "#811E9F": (129, 30, 159), "#B44AC0": (180, 74, 192),
    "#E4ABFF": (228, 171, 255), "#DE107F": (222, 16, 127), "#FF3881": (255, 56, 129), "#FF99AA": (255, 153, 170),
    "#6D482F": (109, 72, 47), "#9C6926": (156, 105, 38), "#FFB470": (255, 180, 112), "#000000": (0, 0, 0),
    "#515252": (81, 82, 82), "#898D90": (137, 141, 144), "#D4D7D9": (212, 215, 217), "#FFFFFF": (255, 255, 255)
}

def get_snapshot_at_time(target_time_str):
    print(f"--- Generating 3D Canvas Snapshot for: {target_time_str} ---")
    start_t = time.perf_counter()

    # Parse Target Timestamp
    try:
        # Append " UTC" to match the CSV format if it's missing
        if "UTC" not in target_time_str:
            target_time_str += " UTC"
    except Exception:
        print("Invalid time format. Use 'YYYY-MM-DD HH:MM:SS'")
        return

    # Lazy Scan & Filter by Time
    lf = pl.scan_csv(DATA_FILE_PATH)
    
    # Filter: Keep only rows BEFORE the target time
    # Note: Do string comparison on the timestamp column. 
    # For 'YYYY-MM-DD...' format, lexicographical sort works perfectly and is faster than parsing dates.
    filtered = lf.filter(pl.col("timestamp") <= target_time_str)

    # Handle Coordinates (Standard & Mod Rectangles)
    filtered = filtered.with_columns(
        pl.col("coordinate").str.split(",").alias("coords_list")
    ).with_columns(
        pl.col("coords_list").list.len().alias("c_count")
    )

    # Branch A: Standard Pixels
    std = filtered.filter(pl.col("c_count") == 2).select([
        pl.col("coords_list").list.get(0).cast(pl.Int16).alias("x"),
        pl.col("coords_list").list.get(1).cast(pl.Int16).alias("y"),
        pl.col("pixel_color")
    ])

    # Branch B: Mod Rectangles (Expansion)
    mod = filtered.filter(pl.col("c_count") == 4).select([
        pl.col("coords_list").list.get(0).cast(pl.Int32).alias("x1"),
        pl.col("coords_list").list.get(1).cast(pl.Int32).alias("y1"),
        pl.col("coords_list").list.get(2).cast(pl.Int32).alias("x2"),
        pl.col("coords_list").list.get(3).cast(pl.Int32).alias("y2"),
        pl.col("pixel_color")
    ]).with_columns([
        pl.int_ranges(pl.col("x1"), pl.col("x2") + 1).alias("x_range"),
        pl.int_ranges(pl.col("y1"), pl.col("y2") + 1).alias("y_range")
    ]).explode("x_range").explode("y_range").select([
        pl.col("x_range").cast(pl.Int16).alias("x"),
        pl.col("y_range").cast(pl.Int16).alias("y"),
        pl.col("pixel_color")
    ])

    combined = pl.concat([std, mod])

    # Aggregation: Get HEIGHT and FINAL COLOR per pixel
    # Maintain order so the "last" color in the CSV is the visible color
    print("  > Aggregating pixel history...")
    
    # Collect first to utilize Polars' efficient group_by context
    # Aggregation Strategy:
    # 1. Height = count()
    # 2. Visible Color = last()
    final_state = (
        combined
        .group_by(["x", "y"])
        .agg([
            pl.len().alias("height"),
            pl.col("pixel_color").last().alias("final_hex")
        ])
        .collect()
    )

    print(f"  > Data processed in {time.perf_counter() - start_t:.2f}s. Points to render: {final_state.height}")

    # --- Constructing the Visualization ---
    print("  > Building 3D Geometry...")

    # 1. Prepare Grid Arrays
    # Initialize Height Grid (Z-axis)
    height_grid = np.zeros((RPLACE_WIDTH, RPLACE_HEIGHT))
    
    # Initialize Color Grid (RGB channels) - Default to White (255,255,255)
    color_grid = np.full((RPLACE_WIDTH, RPLACE_HEIGHT, 3), 255, dtype=np.uint8)

    # Filter Valid Bounds
    valid = final_state.filter(
        (pl.col("x") < RPLACE_WIDTH) & (pl.col("y") < RPLACE_HEIGHT)
    )

    # 2. Map Data to Arrays
    xs = valid["x"].to_numpy()
    ys = valid["y"].to_numpy()
    zs = valid["height"].to_numpy()
    hexes = valid["final_hex"].to_list()

    # Set Heights
    height_grid[xs, ys] = zs

    # Set Colors (Convert Hex to RGB on the fly)
    # Using a loop here is efficient enough for 4M pixels compared to mesh generation
    # Vectorizing this map lookup is possible but complex for little gain in this specific step
    rgb_values = np.array([HEX_TO_RGB.get(h, (255,255,255)) for h in hexes], dtype=np.uint8)
    color_grid[xs, ys] = rgb_values

    # 3. Create PyVista Mesh
    # Use ImageData (formerly UniformGrid)
    grid = pv.ImageData(dimensions=(RPLACE_WIDTH, RPLACE_HEIGHT, 1))
    
    # Assign RGB colors to the mesh
    # Flatten: PyVista expects (N_points, 3) for RGB
    # Note: We transpose (.T) grid values to match VTK's column-major ordering
    grid.point_data["RGB"] = color_grid.swapaxes(0, 1).reshape(-1, 3)
    
    # Assign Height scalar
    grid.point_data["Elevation"] = height_grid.T.flatten()

    # 4. Warp by Scalar (Create the 3D relief)
    # factor=0.5 makes the mountains visible but not infinitely tall, 0.05 makes them more viewable
    warped = grid.warp_by_scalar("Elevation", factor=0.05)

    # --- Rendering ---
    plotter = pv.Plotter()
    
    # Add the mesh. 
    # rgb=True tells PyVista to use the "RGB" array we attached for coloring
    plotter.add_mesh(warped, rgb=True, show_scalar_bar=False)
    
    # Add Height Scale Bar manually for reference
    plotter.add_scalar_bar(title="Placement Count (Height)", mapper=None) # Mapper auto-attaches to active scalars
    
    # Setup Scene
    plotter.show_grid()
    plotter.add_text(f"r/place Snapshot: {target_time_str}", position='upper_left', font_size=12)
    plotter.camera_position = 'iso'
    
    print("  > Opening Interactive Window... (Controls: Left Click=Rotate, Scroll=Zoom)")
    plotter.show()

if __name__ == "__main__":
    # Allow command line argument for time
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = DEFAULT_TIME
        print(f"No time provided. Defaulting to end of event: {target}")
        print("Usage: python rplace_3d_snapshot.py '2022-04-03 12:00:00'")
    
    get_snapshot_at_time(target)