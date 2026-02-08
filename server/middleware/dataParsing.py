import json
import os
import time
import statistics
from constants import *

def readData(state, data_path="data.json"):
    """
    Reads analyzer output incrementally.
    Performs initial calibration if user_profile.json does not exist.
    """

    if not os.path.exists(data_path):
        return

    calibration_mode = not os.path.exists(state.profile_path)

    # Temporary storage for calibration
    calib_buffer = {
        "pitch": [],
        "yaw": [],
        "visual": []
    }

    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i < state.last_line_processed:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # do NOT advance cursor

            metrics = data.get("metrics")
            flags = data.get("flags")

            if not metrics or not flags:
                continue  # malformed line, skip safely

            # --- parse metrics (same as before) ---
            pitch = metrics.get("pitch_engagement", {})
            yaw = metrics.get("yaw_engagement", {})
            visual = metrics.get("visual_stability", {})

            state.metrics["pitch_engagement"].update(pitch)
            state.metrics["yaw_engagement"].update(yaw)
            state.metrics["visual_stability"].update(visual)

            state.flags["phone_checking_mode"] = flags.get("phone_checking_mode", False)

            # calibration buffer logic here...
            if calibration_mode:
                if "mean" in pitch:
                    calib_buffer["pitch"].append(pitch["mean"])
                if "mean" in yaw:
                    calib_buffer["yaw"].append(yaw["mean"])
                if "mean" in visual:
                    calib_buffer["visual"].append(visual["mean"])


            state.last_line_processed += 1




    # --- Perform initial calibration ---
    if calibration_mode:
        
        thresholds = {}
        weights = state.weights.copy()

        for key, values in calib_buffer.items():
            if len(values) < MIN_CALIBRATION_SAMPLES:
                return  # Not enough data yet

            mu = statistics.mean(values)
            sigma = statistics.pstdev(values)

            thresholds[key] = max(0.0, mu - CALIBRATION_K * sigma)

        profile = {
            "thresholds": thresholds,
            "weights": weights,
            "false_alarm_history": {
                "pitch": 0,
                "yaw": 0,
                "visual": 0 
            },
            "good_behavior_counter": 0,
            "long_term_variance": {},
            "last_variance_update_time": time.time(),
            "last_line_processed": state.last_line_processed
        }

        with open(state.profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        print("✅ Initial calibration complete. user_profile.json created.")
