import cv2
import mediapipe as mp

class VisionTracker:
    def __init__(self):
        # 初始化 MediaPipe 人脸检测
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_target_angle(self, frame):
        """返回目标相对于相机中心的角度 (单位: 度)"""
        results = self.mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        
        # 鼻尖的 Landmark 索引是 1
        nose = results.multi_face_landmarks[0].landmark[1]
        # 简单的线性映射：0.5 是中心(0°)，0.0 是左边缘(-45°)，1.0 是右边缘(45°)
        angle = (nose.x - 0.5) * 90 
        return angle