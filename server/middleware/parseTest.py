import os
import json
import time
from state import State
from dataParsing import readData


TEST_DATA = "parseTest.json"
TEST_PROFILE = "test_user_profile.json"


def write_test_data(lines):
    with open(TEST_DATA, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def append_test_data(lines):
    with open(TEST_DATA, "a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def cleanup():
    for f in [TEST_DATA, TEST_PROFILE]:
        if os.path.exists(f):
            os.remove(f)


def run_test():
    cleanup()

    # --- Step 1: create test data ---
    initial_lines = [
        {
            "timestamp": 1,
            "metrics": {
                "pitch_engagement": {"mean": 0.90},
                "yaw_engagement": {"mean": 0.85},
                "visual_stability": {"mean": 0.75}
            },
            "flags": {"phone_checking_mode": False}
        },
        {
            "timestamp": 2,
            "metrics": {
                "pitch_engagement": {"mean": 0.92},
                "yaw_engagement": {"mean": 0.87},
                "visual_stability": {"mean": 0.77}
            },
            "flags": {"phone_checking_mode": False}
        }
    ]

    write_test_data(initial_lines)

    # --- Step 2: run parser (first call triggers calibration) ---
    state = State(profile_path=TEST_PROFILE)
    readData(state, TEST_DATA)

    # BEFORE calibration resets anything, check last_line_processed
    assert state.last_line_processed == 2, "❌ Did not process 2 lines"

    # Check that metrics reflect the LAST parsed line (0.92)
    print(state.metrics)
    print("Profile path:", state.profile_path)

    assert state.metrics["pitch_engagement"]["mean"] == 0.92

    print("✅ Initial parse passed")

    # --- Step 3: calibration should have created the profile ---

    assert os.path.exists(state.profile_path) #"❌ user_profile.json not created"

    with open(state.profile_path) as f:
        profile = json.load(f)

    assert "thresholds" in profile
    assert "pitch" in profile["thresholds"]

    print("✅ Calibration profile created")

    # --- Step 4: append new data ---
    new_lines = [
        {
            "timestamp": 3,
            "metrics": {
                "pitch_engagement": {"mean": 0.91},
                "yaw_engagement": {"mean": 0.86},
                "visual_stability": {"mean": 0.76}
            },
            "flags": {"phone_checking_mode": False}
        }
    ]

    append_test_data(new_lines)

    # --- Step 5: run parser again (incremental read) ---
    readData(state, TEST_DATA)

    assert state.last_line_processed == 3, "❌ Re-read old data"
    assert state.metrics["pitch_engagement"]["mean"] == 0.91
    assert state.flags["phone_checking_mode"] is False

    print("✅ Incremental parse passed")

    print("\n🎉 ALL TESTS PASSED\n")


if __name__ == "__main__":
    run_test()
