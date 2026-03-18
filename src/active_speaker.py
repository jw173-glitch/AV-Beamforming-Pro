import cv2
import mediapipe as mp
import numpy as np


class ActiveSpeakerDetector:
    def __init__(self, threshold=0.01):
        self.threshold = threshold

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        self.prev_values = []

    def is_speaking(self, frame):
        """原有方法，完全不变"""
        results = self.face_mesh.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0].landmark

        upper = landmarks[13]
        lower = landmarks[14]

        mouth_open = abs(upper.y - lower.y)

        self.prev_values.append(mouth_open)
        if len(self.prev_values) > 5:
            self.prev_values.pop(0)

        variation = np.std(self.prev_values)

        if variation > self.threshold:
            return True
        else:
            return False

    # ===== 新增方法 =====

    def is_speaking_with_conf(self, frame):
        """
        在 is_speaking 基础上额外返回置信度 [0.0, 1.0]
        供 fusion.py 软融合使用
        """
        results = self.face_mesh.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        if not results.multi_face_landmarks:
            return False, 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        upper = landmarks[13]
        lower = landmarks[14]
        mouth_open = abs(upper.y - lower.y)

        self.prev_values.append(mouth_open)
        if len(self.prev_values) > 10:  # 窗口从5改为10，更稳定
            self.prev_values.pop(0)

        variation = np.std(self.prev_values)
        is_spk = variation > self.threshold
        conf = float(np.clip(variation / (self.threshold * 2), 0.0, 1.0))

        return is_spk, conf