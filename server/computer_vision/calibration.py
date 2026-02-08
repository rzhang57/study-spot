import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np
import random
from utils import BiometricUtils

# CONFIG
DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

class CalibrationGame:
    def __init__(self):
        # Initialize Camera
        self.cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened(): 
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        self.window_name = "Neuro-Calibration"
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.h, self.w = 720, 1280 

    def draw_instruction_screen(self, frame, title, lines, color=(0, 255, 0)):
        """Renders a clean instruction overlay."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, title, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (50, 180 + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, "Press [SPACE] to Continue", (50, self.h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    def draw_target(self, frame, x, y, time_left, total_time, color=(0, 255, 0)):
        """Draws a smooth target with an anti-aliased shrinking ring."""
        cv2.circle(frame, (x, y), 12, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 15, (255, 255, 255), 2, cv2.LINE_AA)

        progress = max(0, min(1, time_left / total_time))
        ring_radius = int(15 + (65 * progress))
        cv2.circle(frame, (x, y), ring_radius, (200, 200, 200), 1, cv2.LINE_AA)

    def run_phase(self, phase_name, targets, duration_per_target=1.5):
        csv_path = os.path.join(DATA_DIR, f"calibration_{phase_name}.csv")
        f = open(csv_path, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['rel_time', 'target_x', 'target_y', 'gaze_x', 'gaze_y', 'pitch', 'yaw', 'ear'])

        start_time = time.time()
        target_idx = 0
        last_switch = start_time
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success: break
                
                frame = cv2.flip(frame, 1)
                self.h, self.w, _ = frame.shape
                curr_time = time.time()
                time_since_switch = curr_time - last_switch

                if time_since_switch > duration_per_target:
                    target_idx += 1
                    last_switch = curr_time
                    time_since_switch = 0
                
                if target_idx >= len(targets): break

                tx, ty = targets[target_idx]
                screen_x, screen_y = int(tx * self.w), int(ty * self.h)
                
                time_left = duration_per_target - time_since_switch
                self.draw_target(frame, screen_x, screen_y, time_left, duration_per_target)
                
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    ear = BiometricUtils.calculate_ear(lm)
                    gaze_x, gaze_y = BiometricUtils.get_gaze_coords(lm)
                    pitch, yaw = BiometricUtils.get_head_pose(lm, self.w, self.h)
                    writer.writerow([curr_time - start_time, tx, ty, gaze_x, gaze_y, pitch, yaw, ear])

                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): return False
        finally:
            f.close()
        return True

    def wait_for_user(self, title, lines, color=(0, 255, 0)):
        while True:
            success, frame = self.cap.read()
            if not success: return False
            frame = cv2.flip(frame, 1)
            self.draw_instruction_screen(frame, title, lines, color)
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): return True
            if key == ord('q'): return False

    def run(self):
        try:
            # PHASE 1: Convergent
            p1_info = [
                "We are setting your 'Focus Boundaries'.",
                "1. Follow the green dot to 9 points on your screen.",
                "2. Move your eyes AND head naturally, as if you were studying.",
                "3. Try to keep the same posture you use for deep work."
            ]
            if not self.wait_for_user("PHASE 1: CONVERGENT FOCUS", p1_info): return

            targets_p1 = [
                (0.5, 0.5), (0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95),
                (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5), (0.5, 0.5)
            ]
            self.run_phase("convergent", targets_p1, duration_per_target=1.5)

            # PHASE 2: Divergent
            p2_info = [
                "We are measuring your 'Search Velocity'.",
                "1. The dot will move quickly and randomly.",
                "2. Track it as fast as possible with your eyes.",
                "3. This helps us distinguish between searching and distraction."
            ]
            if not self.wait_for_user("PHASE 2: DIVERGENT SCANNING", p2_info): return

            targets_p2 = [(random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)) for _ in range(30)]
            self.run_phase("divergent", targets_p2, duration_per_target=0.7)

            # NEW: PHASE 3: Calibration Complete
            final_info = [
                "Your focus baseline has been successfully captured.",
                "1. Convergent map saved to: data/calibration_convergent.csv",
                "2. Divergent map saved to: data/calibration_divergent.csv",
            ]
            self.wait_for_user("CALIBRATION COMPLETE!", final_info, color=(0, 255, 255))

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    CalibrationGame().run()