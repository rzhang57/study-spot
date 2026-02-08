# -----------------------------
# Defaults
# -----------------------------
#TODO: change the default values and all the names
default_weights = {
    "pitch": 0.35,
    "yaw": 0.25,
    "visual": 0.20,
}

#TODO: ADJUST THESE FOR SURE! and we only need this if we aren't doing the calibration
# step
default_thresholds = {
    "max_offscreen_time": 3.0,
    "blink_rate_threshold": 0.60,
    "head_movement_tolerance": 0.30,
    "expression_drop_threshold": 0.20,
}

default_false_alarm_history = {
    "gaze": 0,
    "head": 0,
    "blink": 0,
    "expression": 0,
}

default_long_term_variance = {
    "gaze": None,
    "head": None,
    "blink": None,
    "expression": None,
}

default_good_behavior = {
    "gaze": 0,
    "head": 0,
    "blink": 0,
    "expression": 0,
}



# -----------------------------
# False alarm logic
# -----------------------------
FALSE_ALARM_LIMIT = 3

# -----------------------------
# Variance thresholds
# -----------------------------
#TODO: adjust these depending on what kind of variance or data we actually get
LOW_VARIANCE_THRESHOLD = 0.15
VARIANCE_IMPROVEMENT_THRESHOLD = 0.05  # recommended for long-term tightening

# -----------------------------
# Weight adjustments
# -----------------------------
#TODO: can talk about and adjust these factors
WEIGHT_DECREMENT_FACTOR = 0.925
MIN_WEIGHT = 0.10


# TODO adjust these names and order and stuff.
THRESHOLD_KEYS = {
    "gaze": "max_offscreen_time",
    "blink": "blink_rate_threshold",
    "head": "head_movement_tolerance",
    "expression": "expression_drop_threshold"
}

# -----------------------------
# Threshold adjustments (relaxing)
# -----------------------------
#TODO: can talk about and adjust these factors
THRESHOLD_LOOSEN = 1.1

# -----------------------------
# Threshold tightening (optional)
# -----------------------------
THRESHOLD_TIGHTEN = 0.975

# -----------------------------
# Engagement thresholds
# -----------------------------
HIGH_ENGAGEMENT_THRESHOLD = 0.75

# -----------------------------
# Good behavior logic
# -----------------------------
GOOD_BEHAVIOR_LIMIT = 60  # e.g., 60 cycles of good behavior


# --- Calibration constants ---
MIN_CALIBRATION_SAMPLES = 1        # ~5–7 seconds of data at typical frame rates
CALIBRATION_K = 1.0                  # 1 standard deviation below mean
