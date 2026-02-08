'''import pandas as pd
import numpy as np
import time
import json
import os
import sys
from collections import deque
from scipy.spatial import distance

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.getcwd(), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")
# Default 'Z-Score' Thresholds (How many Standard Deviations before we flag it)
DEFAULT_CONFIG = {
    "yaw_z_threshold": 2.0,      # Strict: 2.0 deviations (Screen is narrow horizontally)
    "pitch_z_threshold": 3.0,    # Lenient: 3.0 deviations (Looking down is common)
    "blink_z_threshold": 2.5,    # For detecting rapid blinking
    "gaze_velocity_threshold": 1.5 # Multiplier of calibration max velocity
}

class NeuroAnalyzer:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.calib_mean = {} # Stores Mean (μ)
        self.calib_std = {}  # Stores Std Dev (σ)
        self.max_saccade_velocity = 0.0 # From Divergent Calibration
        
        # --- ROLLING BUFFERS ---
        # We hold 30 seconds of data @ ~30 FPS = 900 frames
        # usage: buffer[-1] is newest, buffer[0] is oldest
        self.window_size_30s = 900 
        self.buffer = deque(maxlen=self.window_size_30s)
        
        # Load the baseline 'Normal'
        if not self.load_calibration():
            print("CRITICAL: Calibration files not found. Run calibration.py first.")
            sys.exit(1)

    def load_calibration(self):
        """
        Loads the 'Convergent' and 'Divergent' CSVs to establish baselines.
        Calculates μ and σ for Pitch, Yaw, EAR, and Gaze.
        """
        conv_path = os.path.join(DATA_DIR, "calibration_convergent.csv")
        div_path = os.path.join(DATA_DIR, "calibration_divergent.csv")

        if not os.path.exists(conv_path) or not os.path.exists(div_path):
            return False

        try:
            # 1. Process Convergent (Static Focus)
            df_conv = pd.read_csv(conv_path)
            metrics = ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']
            
            for m in metrics:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = df_conv[m].std()
                # Safety: Avoid divide-by-zero if variance is 0
                if self.calib_std[m] == 0: self.calib_std[m] = 0.001

            # 2. Process Divergent (Dynamic Speed)
            df_div = pd.read_csv(div_path)
            # Calculate raw velocity between frames: sqrt((x2-x1)^2 + (y2-y1)^2)
            gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
            velocities = np.linalg.norm(gaze_diffs, axis=1)
            # We take the 95th percentile as "Max Normal Speed" to ignore glitches
            self.max_saccade_velocity = np.percentile(velocities, 95)
            
            print(f"✅ Calibration Loaded. Baseline Pitch: {self.calib_mean['pitch']:.2f} (σ={self.calib_std['pitch']:.2f})")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def check_feedback_updates(self):
        """
        Checks if 'threshold_feedback.json' exists and updates thresholds.
        This allows the downstream game/app to make the analyzer 'stricter' or 'nicer'.
        """
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    # Only update valid keys
                    for k, v in updates.items():
                        if k in self.config:
                            self.config[k] = float(v)
                    # Optional: Delete file after reading so we don't re-read old commands
                    # os.remove(FEEDBACK_FILE) 
            except:
                pass # Ignore file read errors (race conditions)

    def get_z_score(self, value, metric):
        """Calculates Standard Score: (X - Mean) / StdDev"""
        return (value - self.calib_mean[metric]) / self.calib_std[metric]

    def analyze_window(self):
        """
        The Core Logic. Processes the current 30s buffer.
        Returns a dictionary of metrics and flags.
        """
        if len(self.buffer) < 30: return None # Wait for data

        # Convert deque to DataFrame for vector operations (Fast)
        df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
        
        # --- TIER 1: GROSS DISTRACTION (30s Window) ---
        # We use the full 30s window for stable posture analysis
        avg_pitch = df['pitch'].mean()
        avg_yaw = df['yaw'].mean()
        
        # Calculate Z-Scores
        pitch_z = self.get_z_score(avg_pitch, 'pitch')
        yaw_z = abs(self.get_z_score(avg_yaw, 'yaw')) # Direction doesn't matter for Yaw deviation
        
        # 1. Yaw Violation (Looking Left/Right)
        # We use a STRICT threshold here because the screen is finite.
        is_yaw_bad = yaw_z > self.config['yaw_z_threshold']
        
        # 2. Pitch Violation (Looking Up/Down)
        # We check if pitch is significantly LOWER (looking down) or HIGHER (looking up)
        # Note: Depending on calibration, ensure 'looking down' moves Z in the right direction. 
        # Here we assume deviation in EITHER direction is bad.
        is_pitch_bad = abs(pitch_z) > self.config['pitch_z_threshold']
        
        # 3. The "Typing Veto"
        # If user is typing (> 0 keys in last 5 seconds), they are NOT distracted.
        # They are likely looking at the keyboard or notes.
        recent_keys = df['keys'].tail(150).sum() # Last ~5 seconds
        is_typing = recent_keys > 0
        
        # Final Distraction Flag
        is_distracted = (is_yaw_bad or is_pitch_bad) and (not is_typing)

        # --- TIER 2: COGNITIVE STATE (10s Window) ---
        # We only analyze this if they are ostensibly looking at the screen.
        # Slice the last 10 seconds (approx 300 frames)
        df_10s = df.tail(300)
        
        focus_score = 0.0
        stim_score = 0.0
        stress_score = 0.0
        
        blink_stats = {"interval_median": 0.0, "interval_variance": 0.0}

        if not is_distracted:
            # A. Calculate Gaze Dynamics
            # Velocity: Distance between consecutive frames
            gaze_diffs = np.diff(df_10s[['gaze_x', 'gaze_y']].values, axis=0)
            gaze_velocity = np.linalg.norm(gaze_diffs, axis=1).mean()
            
            # Dispersion: Standard Deviation of position
            gaze_variance = df_10s['gaze_x'].std() + df_10s['gaze_y'].std()
            
            # B. Calculate Blink Intervals
            # We assume a blink is when EAR drops below (Mean - 2*Std)
            blink_thresh = self.calib_mean['ear'] - (2 * self.calib_std['ear'])
            # Boolean mask of "Is Blinking"
            blink_frames = df_10s[df_10s['ear'] < blink_thresh].index.tolist()
            
            if len(blink_frames) > 1:
                # Calculate time between blinks (in frames, converted to seconds approx)
                intervals = np.diff(blink_frames) / 30.0 # Assuming ~30fps
                blink_stats["interval_median"] = np.median(intervals)
                blink_stats["interval_variance"] = np.var(intervals)
            
            # C. Calculate Mouth Activity
            mouth_activity = df_10s['mouth'].mean()
            mouth_z = (mouth_activity - 0.0) / 0.05 # Approximate normalization
            
            # --- SCORING LOGIC (The "Mixing Board") ---
            
            # 1. Focus Score (Flow)
            # Conditions: Low Gaze Variance + Low Mouth + Low Blink Rate
            # We normalize these inversely (Lower is Better)
            focus_score = (1.0 / (1.0 + gaze_variance)) * 0.5 + \
                          (1.0 / (1.0 + mouth_activity)) * 0.3
            
            # 2. Stimming Score (ADHD Regulation)
            # Conditions: Low Gaze Variance (Still working) + HIGH Mouth (Chewing/Biting)
            stim_score = (1.0 / (1.0 + gaze_variance)) * 0.4 + \
                         min(1.0, mouth_activity * 10) * 0.6 # Boost mouth signal
            
            # 3. Stress Score (Overload)
            # Conditions: High Gaze Velocity (Chaos) + High Blink Rate (Startle)
            # Use max_saccade_velocity from calibration as the benchmark
            norm_velocity = min(1.0, gaze_velocity / (self.max_saccade_velocity + 0.0001))
            stress_score = norm_velocity * 0.6 + \
                           (1.0 if len(blink_frames) > 5 else 0.0) * 0.4 # Penalty for rapid blinking

        return {
            "timestamp": time.time(),
            "flags": {
                "is_distracted": bool(is_distracted),
                "is_typing_veto": bool(is_typing),
                "pitch_violation": bool(is_pitch_bad),
                "yaw_violation": bool(is_yaw_bad)
            },
            "state_probabilities": {
                "focus": round(focus_score, 2),
                "stim": round(stim_score, 2),
                "stress": round(stress_score, 2)
            },
            "biometrics": {
                "pitch_z": round(pitch_z, 2),
                "yaw_z": round(yaw_z, 2),
                "blink_interval_median": round(blink_stats["interval_median"], 2)
            }
        }

    def run(self):
        print("🧠 NeuroAnalyzer Initialized. Waiting for Watcher data...")
        
        # 1. Find the latest telemetry file
        while True:
            files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
            if files:
                current_file = os.path.join(DATA_DIR, files[-1])
                print(f"📂 Tracking: {current_file}")
                break
            time.sleep(1)

        # 2. Main Loop
        with open(current_file, 'r') as f:
            # Skip header
            f.readline()
            
            while True:
                # A. Check for external feedback (Crosstalk)
                self.check_feedback_updates()
                
                # B. Read new lines
                line = f.readline()
                if not line:
                    time.sleep(0.1) # Wait for Watcher to write
                    continue
                
                # C. Parse & Buffer
                try:
                    # CSV: rel_time, ear, gaze_x, gaze_y, mouth_dist, pitch, yaw, keys, mouse
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    
                    data_point = [float(x) for x in parts]
                    self.buffer.append(data_point)
                    
                    # D. Analyze (Every ~30 frames / 1 sec to save CPU)
                    if len(self.buffer) % 30 == 0:
                        result = self.analyze_window()
                        if result:
                            # E. Output JSON to Stdout (For friend's backend)
                            print(json.dumps(result))
                            sys.stdout.flush() # Ensure immediate sending
                            
                except ValueError:
                    continue

if __name__ == "__main__":
    NeuroAnalyzer().run()'''

#Version 2

import pandas as pd
import numpy as np
import time
import json
import os
import sys
from collections import deque

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.getcwd(), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "threshold_feedback.json")

# Default Thresholds for Z-Score deviations
DEFAULT_CONFIG = {
    "yaw_sigma_tolerance": 2.0,    # Horizontal tolerance
    "pitch_sigma_tolerance": 3.0,  # Vertical tolerance
    "blink_sigma_tolerance": 2.5   # Blink rate tolerance
}

class NeuroAnalyzer:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.calib_mean = {}
        self.calib_std = {}
        self.max_saccade_velocity = 0.0
        
        # Buffer: 10 seconds of data @ 30 FPS = 300 frames
        # We focus on the immediate window for engagement scoring
        self.window_size = 300
        self.buffer = deque(maxlen=self.window_size)
        
        if not self.load_calibration():
            print("CRITICAL: Calibration files not found. Run calibration.py first.")
            sys.exit(1)

    def load_calibration(self):
        """Loads baseline μ and σ from calibration files."""
        conv_path = os.path.join(DATA_DIR, "calibration_convergent.csv")
        div_path = os.path.join(DATA_DIR, "calibration_divergent.csv")

        if not os.path.exists(conv_path) or not os.path.exists(div_path): return False

        try:
            # 1. Convergent Baseline (Static Focus)
            df_conv = pd.read_csv(conv_path)
            for m in ['pitch', 'yaw', 'ear', 'gaze_x', 'gaze_y']:
                self.calib_mean[m] = df_conv[m].mean()
                self.calib_std[m] = df_conv[m].std()
                if self.calib_std[m] == 0: self.calib_std[m] = 0.001

            # 2. Divergent Baseline (Max Speed)
            df_div = pd.read_csv(div_path)
            gaze_diffs = np.diff(df_div[['gaze_x', 'gaze_y']].values, axis=0)
            velocities = np.linalg.norm(gaze_diffs, axis=1)
            self.max_saccade_velocity = np.percentile(velocities, 95)
            
            print(f"✅ Calibration Loaded. Baseline Pitch: {self.calib_mean['pitch']:.2f}")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def check_feedback_updates(self):
        """Reads external JSON to adjust thresholds dynamically."""
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    updates = json.load(f)
                    for k, v in updates.items():
                        if k in self.config: self.config[k] = float(v)
            except: pass

    def get_gaussian_score(self, value, metric, tolerance_sigma):
        """
        Converts a raw value into an 'Engagement Probability' (0.0 to 1.0).
        Uses a Gaussian curve centered on the calibrated mean.
        
        Logic:
        - If Value == Mean, Score = 1.0 (Perfect Engagement)
        - If Value deviates by 'tolerance_sigma', Score drops significantly.
        """
        mu = self.calib_mean[metric]
        sigma = self.calib_std[metric]
        
        # Calculate Z-distance
        z = abs(value - mu) / sigma
        
        # Gaussian Decay: e^(-(z^2) / (2 * tolerance^2))
        # This creates a bell curve where 0 deviation = 1.0 score
        score = np.exp(-(z**2) / (2 * (tolerance_sigma**2)))
        return float(score)

    def calculate_window_stats(self, values):
        """
        Calculates the requested statistical format for a list of values.
        Returns: {mean, std_dev, n, M2}
        """
        arr = np.array(values)
        n = len(arr)
        if n == 0: return {"mean": 0.0, "std_dev": 0.0, "n": 0, "M2": 0.0}
        
        mean = np.mean(arr)
        # M2 is the sum of squared deviations from the mean
        m2 = np.sum((arr - mean)**2)
        std_dev = np.sqrt(m2 / n)
        
        return {
            "mean": round(float(mean), 4),
            "std_dev": round(float(std_dev), 4),
            "n": int(n),
            "M2": round(float(m2), 4)
        }

    def analyze_window(self):
        if len(self.buffer) < 10: return None

        # Convert to DataFrame
        df = pd.DataFrame(list(self.buffer), columns=['time', 'ear', 'gaze_x', 'gaze_y', 'mouth', 'pitch', 'yaw', 'keys', 'mouse'])
        
        # --- 1. CALCULATE RAW SCORES (0.0 - 1.0) PER FRAME ---
        # We apply the Gaussian scoring to every single frame in the buffer
        # to get a vector of "Momentary Engagement"
        
        # Pitch Score (Vertical Engagement)
        pitch_scores = df['pitch'].apply(lambda x: self.get_gaussian_score(x, 'pitch', self.config['pitch_sigma_tolerance']))
        
        # Yaw Score (Horizontal Engagement)
        yaw_scores = df['yaw'].apply(lambda x: self.get_gaussian_score(x, 'yaw', self.config['yaw_sigma_tolerance']))
        
        # Gaze Score (Stability)
        # For gaze, we assume the calibrated MEAN is the "center of work".
        # This is an approximation. Alternatively, we could score based on velocity (Lower velocity = Higher engagement)
        gaze_diffs = np.diff(df[['gaze_x', 'gaze_y']].values, axis=0)
        gaze_velocity = np.linalg.norm(gaze_diffs, axis=1)
        # Normalize velocity score: 1.0 if stationary, 0.0 if moving at max saccade speed
        velocity_scores = np.clip(1.0 - (gaze_velocity / (self.max_saccade_velocity + 1e-6)), 0.0, 1.0)

        # --- 2. CALCULATE STATISTICS ---
        # We now pack these score arrays into the requested format
        stats_output = {
            "pitch_engagement": self.calculate_window_stats(pitch_scores),
            "yaw_engagement": self.calculate_window_stats(yaw_scores),
            "visual_stability": self.calculate_window_stats(velocity_scores),
        }

        # --- 3. BOOLEAN FLAGS ---
        
        # Activity Check (Last 1 second)
        recent_keys = df['keys'].tail(30).sum()
        recent_mouse = df['mouse'].tail(30).sum()
        
        is_keyboard_active = bool(recent_keys > 0)
        is_mouse_active = bool(recent_mouse > 0)


        
        avg_pitch_score = stats_output["pitch_engagement"]["mean"]
        
        # EAR Check: Is it significantly below baseline?
        # We check if the current window mean is < (Baseline - 2*Sigma)
        current_ear_mean = df['ear'].mean()
        ear_threshold = self.calib_mean['ear'] - (2 * self.calib_std['ear'])
        is_ear_low = current_ear_mean < ear_threshold


        return {
            "timestamp": time.time(),
            "metrics": stats_output,
            "flags": {
                "mouse_movement": is_mouse_active,
                "keyboard_stroke": is_keyboard_active
            }
        }

    def run(self):
        print("🧠 NeuroAnalyzer Active. Outputting Statistical JSON...")
        
        # Find latest file
        while True:
            files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("telemetry_")])
            if files:
                current_file = os.path.join(DATA_DIR, files[-1])
                break
            time.sleep(1)

        with open(current_file, 'r') as f:
            f.readline() # Header
            while True:
                self.check_feedback_updates()
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 9: continue
                    self.buffer.append([float(x) for x in parts])
                    
                    # Analyze every 10 frames (approx 3 times/sec)
                    if len(self.buffer) % 10 == 0:
                        result = self.analyze_window()
                        if result:
                            print(json.dumps(result))
                            sys.stdout.flush()
                except ValueError: continue

if __name__ == "__main__":
    NeuroAnalyzer().run()