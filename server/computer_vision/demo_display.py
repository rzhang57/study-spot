import cv2
import json
import os
import time
import numpy as np

# CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(SCRIPT_DIR, "data", "neuro_output.json")

def run_demo():
    print("🚀 OPENCV DEMO ACTIVE. Press 'q' to quit.")
    
    while True:
        bg_color = (0, 0, 0) # Black default
        status_text = "WAITING..."
        
        try:
            if os.path.exists(JSON_FILE):
                with open(JSON_FILE, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        data = json.loads(lines[-1])
                        is_phone = data['flags']['phone_checking_mode']
                        focus = data['state_probabilities']['focus']
                        
                        if is_phone:
                            bg_color = (0, 0, 255) # RED (BGR)
                            status_text = "DISTRACTED"
                        else:
                            bg_color = (0, 255, 0) # GREEN (BGR)
                            status_text = f"FOCUSED: {focus}"
        except:
            pass

        # Create a simple window
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img[:] = bg_color
        
        cv2.putText(img, status_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.imshow("NEURO-LINK STATUS", img)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()