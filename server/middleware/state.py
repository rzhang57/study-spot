import json
import os
import time
from constants import *

class State:
    def __init__(self, profile_path="user_profile.json"):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.profile_path = os.path.join(base_dir, profile_path)

        # -----------------------------
        # Load persistent profile
        # -----------------------------
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r") as f:
                profile = json.load(f)
        else:
            profile = {}

        # -----------------------------
        # Initialize WEIGHTS
        # -----------------------------
        self.weights = {
            k: profile.get("weights", {}).get(k, default_weights[k])
            for k in default_weights
        }

        # -----------------------------
        # Initialize THRESHOLDS
        # -----------------------------
        self.thresholds = {
            k: profile.get("thresholds", {}).get(k, default_thresholds[k])
            for k in default_thresholds
        }

        # -----------------------------
        # False alarm counters
        # -----------------------------
        self.false_alarm_history = profile.get(
            "false_alarm_history", default_false_alarm_history
        )

        # -----------------------------
        # Good behavior counters
        # -----------------------------
        self.good_behavior_counter = profile.get(
            "good_behavior_counter", default_good_behavior
        )

        # -----------------------------
        # Long-term variance snapshots
        # -----------------------------
        self.long_term_variance = profile.get(
            "long_term_variance", default_long_term_variance
        )

        # -----------------------------
        # Timestamp for variance updates
        # -----------------------------
        self.last_variance_update_time = profile.get(
            "last_variance_update_time", time.time()
        )

        # =====================================================
        # SESSION-ONLY STATE (MATCHES ANALYZER JSON EXACTLY)
        # =====================================================

        self.metrics = {
            "pitch_engagement": {
                "mean": None,
                "std_dev": None,
                "n": 0,
                "M2": 0.0,
            },
            "yaw_engagement": {
                "mean": None,
                "std_dev": None,
                "n": 0,
                "M2": 0.0,
            },
            "visual_stability": {
                "mean": None,
                "std_dev": None,
                "n": 0,
                "M2": 0.0,
            },
        }

        self.flags = {
            "phone_checking_mode": profile.get("phone_checking_mode", False),
        }

        # -----------------------------
        # Stream bookkeeping
        # -----------------------------
        self.last_line_processed = profile.get("last_line_processed", 0)

        


    # Save persistent profile
    # -----------------------------
    def save_profile(self):
        profile = {
            "weights": self.weights,
            "thresholds": self.thresholds,
            "false_alarm_history": self.false_alarm_history,
            "long_term_variance": self.long_term_variance,
            "good_behavior_counter": self.good_behavior_counter,
            "last_variance_update_time": self.last_variance_update_time,
            "last_line_processed": self.last_line_processed
        }
        with open(self.profile_path, "w") as f:
            json.dump(profile, f, indent=2)
