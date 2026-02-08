#This file contains code of the biometric utils class so that it can be imported by both the watcher and calibrator 

import numpy as np
import cv2

# ==========================================
# SHARED BIOMETRIC MATHEMATICS
# ==========================================
class BiometricUtils:
    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]
    IRIS_L_IDX = 473
    IRIS_R_IDX = 468
    MOUTH_TOP_OUTER = 0
    MOUTH_BOTTOM_OUTER = 17
    
    # Landmarks for 2D Pitch/Yaw
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    @staticmethod
    def calculate_ear(landmarks):
        def get_single_ear(indices):
            v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
            v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
            h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
            return (v1 + v2) / (2.0 * h)
        return (get_single_ear(BiometricUtils.L_EYE) + get_single_ear(BiometricUtils.R_EYE)) / 2.0
    
    @staticmethod
    def get_gaze_coords(landmarks):
        l_pt, r_pt = landmarks[BiometricUtils.IRIS_L_IDX], landmarks[BiometricUtils.IRIS_R_IDX]
        return (l_pt.x + r_pt.x) / 2.0, (l_pt.y + r_pt.y) / 2.0

    @staticmethod
    def get_mouth_dist(landmarks):
        return np.abs(landmarks[BiometricUtils.MOUTH_TOP_OUTER].y - landmarks[BiometricUtils.MOUTH_BOTTOM_OUTER].y)

    @staticmethod
    def get_head_pose(landmarks, img_w, img_h):
        """
        Calculates 2D Ratios. Returns Pitch (0-1) and Yaw (Pixel Diff).
        """
        nose = landmarks[BiometricUtils.NOSE_TIP]
        chin = landmarks[BiometricUtils.CHIN]
        l_eye = landmarks[BiometricUtils.LEFT_EYE_OUTER]
        r_eye = landmarks[BiometricUtils.RIGHT_EYE_OUTER]
        
        # Pitch: Ratio of Nose-to-Chin vs Face Height
        eyes_mid_y = (l_eye.y + r_eye.y) / 2.0
        face_height = np.abs(chin.y - eyes_mid_y)
        nose_to_chin = np.abs(chin.y - nose.y)
        pitch_ratio = nose_to_chin / (face_height + 1e-6)
        
        # Yaw: Horizontal Asymmetry (Scaled by Image Width)
        nose_x = nose.x * img_w
        l_eye_x = l_eye.x * img_w
        r_eye_x = r_eye.x * img_w
        yaw_diff = np.abs(nose_x - l_eye_x) - np.abs(nose_x - r_eye_x)
        
        return pitch_ratio, yaw_diff
    




    '''
    Copy of the original class from the finalized wather.py: 
    
    
     ==========================================
# MODULE 3: BIOMETRIC UTILS (STABILIZED 2D)
# ==========================================
class BiometricUtils:
    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]
    IRIS_L_IDX = 473
    IRIS_R_IDX = 468
    MOUTH_TOP_OUTER = 0
    MOUTH_BOTTOM_OUTER = 17
    
    # Landmarks for 2D Pitch/Yaw
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    @staticmethod
    def calculate_ear(landmarks):
        def get_single_ear(indices):
            v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
            v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
            h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
            return (v1 + v2) / (2.0 * h)
        return (get_single_ear(BiometricUtils.L_EYE) + get_single_ear(BiometricUtils.R_EYE)) / 2.0
    
    @staticmethod
    def get_gaze_coords(landmarks):
        l_pt, r_pt = landmarks[BiometricUtils.IRIS_L_IDX], landmarks[BiometricUtils.IRIS_R_IDX]
        return (l_pt.x + r_pt.x) / 2.0, (l_pt.y + r_pt.y) / 2.0

    @staticmethod
    def get_mouth_dist(landmarks):
        return np.abs(landmarks[BiometricUtils.MOUTH_TOP_OUTER].y - landmarks[BiometricUtils.MOUTH_BOTTOM_OUTER].y)

    @staticmethod
    def get_head_pose(landmarks, img_w, img_h):
        """
        Calculates 2D Ratios. Returns Pitch (0-1) and Yaw (Pixel Diff).
        """
        nose = landmarks[BiometricUtils.NOSE_TIP]
        chin = landmarks[BiometricUtils.CHIN]
        l_eye = landmarks[BiometricUtils.LEFT_EYE_OUTER]
        r_eye = landmarks[BiometricUtils.RIGHT_EYE_OUTER]
        
        # Pitch: Ratio of Nose-to-Chin vs Face Height
        eyes_mid_y = (l_eye.y + r_eye.y) / 2.0
        face_height = np.abs(chin.y - eyes_mid_y)
        nose_to_chin = np.abs(chin.y - nose.y)
        pitch_ratio = nose_to_chin / (face_height + 1e-6)
        
        # Yaw: Horizontal Asymmetry (Scaled by Image Width)
        nose_x = nose.x * img_w
        l_eye_x = l_eye.x * img_w
        r_eye_x = r_eye.x * img_w
        yaw_diff = np.abs(nose_x - l_eye_x) - np.abs(nose_x - r_eye_x)
        
        return pitch_ratio, yaw_diff
'''