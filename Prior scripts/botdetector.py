import sys
import polars as p2
import time
from datetime import datetime
import botnet_detector 

DATA_FILE_PATH = 'processed_place_data.parquet'
EVENT_START = '2022-04-01 12:44:10' 

def detect_bots(file_path=DATA_FILE_PATH):
    # "Bots" Must have placed at least 10 pixels to be considered active enough to warrant detection.
    start_perf = time.perf_counter_ns() 

    try:
        lf = p2.scan_parquet(file_path)
        
        # filter out users who have placed less than 10 pixels.
        df_window = lf.filter(
            p2.len().over("user_id_int") > 10
        )

        # Find users who are precisely placing pixels with very little time standard deviation
        # Calculate standard deviation of time intervals between pixels. If < 1s, it's suspicious.
        # The 1 second standard deviation metric is arbitrary, but should catch bots while leaving out the majority of regular humans.
        no_time_deviation = (
            df_window.sort(["user_id_int", "seconds_since_start"])
            .with_columns([
                # Find the time between placements
                (p2.col("seconds_since_start") - p2.col("seconds_since_start").shift(1))
                .over("user_id_int").alias("time_diff")
            ])
            .group_by("user_id_int")
            # calculate the standard deviation in seconds
            .agg(p2.col("time_diff").std().alias("std_dev_seconds"))
            .filter(p2.col("std_dev_seconds") < 2)
            .collect()
        )

        # Find users who are consistently active (at least 1 pixel an hour) for 24 or more consecutive hours
        # Bots need no sleep
        # At least one pixel an hour for 24 consecutive hours may capture some dedicated fans, but should still be considered irregular
        no_sleep = (
            df_window
            # 1. Get unique hours per user
            .select(["user_id_int", (p2.col("seconds_since_start") / 3600).cast(p2.Int32).alias("hour_id")])
            .unique() 
            .sort(["user_id_int", "hour_id"])
            # 2. Identify streaks using a "difference from a sequence" trick
            # If hours are 5, 6, 7... and we subtract 0, 1, 2... the result is always 5.
            # If there is a gap (5, 6, 8), the result changes (5, 5, 6).
            .with_columns(
                (p2.col("hour_id") - p2.col("hour_id").cum_count().over("user_id_int")).alias("streak_id")
            )
            # 3. Count the size of each streak group
            .group_by(["user_id_int", "streak_id"])
            .agg(p2.len().alias("streak_length"))
            # 4. Filter for users who have ANY streak >= 24
            .filter(p2.col("streak_length") >= 24)
            .select("user_id_int")
            .unique()
            .collect()
        )

        # Find users who place pixels linearly, like horizontally or vertically, without deviation
        # Calculate dx/dy. If 95% of moves are pure horizontal (dx=1, dy=0) or vertical, flag them.
        linear_placement = (
            df_window.sort(["user_id_int", "seconds_since_start"])
            .with_columns([
                (p2.col("x") - p2.col("x").shift(1)).over("user_id_int").alias("dx"),
                (p2.col("y") - p2.col("y").shift(1)).over("user_id_int").alias("dy")
            ])
            .with_columns(
                ((p2.col("dx").abs() == 1) & (p2.col("dy") == 0) | 
                 (p2.col("dx") == 0) & (p2.col("dy").abs() == 1))
                .cast(p2.Int32).alias("is_linear")
            )
            .group_by("user_id_int")
            .agg(p2.col("is_linear").mean().alias("linearity_score"))
            .filter(p2.col("linearity_score") > 0.95)
            .collect()
        )

        # Find users who only place pixels within a small area (3x3?)
        # Calculate the Bounding Box (Max X - Min X) and (Max Y - Min Y). 
        # If both dimensions are <= 3, the user is contained in a 3x3 box.
        low_coordinate_deviation = (
            df_window.group_by("user_id_int")
            .agg([
                (p2.col("x").max() - p2.col("x").min()).alias("x_range"),
                (p2.col("y").max() - p2.col("y").min()).alias("y_range")
            ])
            .filter((p2.col("x_range") <= 3) & (p2.col("y_range") <= 3))
            .collect()
        )

        # Find users who only placed a single color.
        # Logic: Users with only 1 unique color_name.
        mono_chromatic = (
            df_window.group_by("user_id_int")
            .agg(p2.col("color_name").n_unique().alias("unique_colors"))
            .filter(p2.col("unique_colors") == 1)
            .collect()
        )


        execution_time_ms = (time.perf_counter_ns() - start_perf) // 1_000_000

        print("Analysis excludes users with fewer than 10 pixel placements")
        print(f"Number of users with little time variance: {no_time_deviation.height}")
        print(f"Number of users with 24+ hours of consecutive activity: {no_sleep.height}")
        print(f"Number of users with strictly linear placements: {linear_placement.height}")
        print(f"Number of users who placed pixels in few coordinates: {low_coordinate_deviation.height}")
        print(f"Number of users who only placed a single color: {mono_chromatic.height}")
        print(f"Execution Time: {execution_time_ms} ms")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    detect_bots()