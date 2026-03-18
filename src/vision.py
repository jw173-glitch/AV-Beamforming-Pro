import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh  # 新增：用于唇部ROI和正面置信度

def get_face_angle(frame, fov=90):
    """原有函数，完全不变"""
    h, w, _ = frame.shape

    with mp_face.FaceDetection(model_selection=0) as detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box

            cx = bbox.xmin + bbox.width / 2
            angle = (cx - 0.5) * fov

            return angle

    return None


# ===== 以下为新增部分 =====

def get_frontal_confidence(angle, max_yaw=45.0):
    """
    根据水平角度估算正面置信度
    angle=0 → conf=1.0（正对）
    angle=±45 → conf=0.0（侧面）
    """
    if angle is None:
        return 0.0
    return float(np.clip(1.0 - abs(angle) / max_yaw, 0.0, 1.0))


def get_lip_roi(frame):
    """
    用 FaceMesh 提取唇部 ROI，供主动说话人检测和后续 AV 模型使用
    返回：lip_roi (numpy array) 或 None
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1
    ) as mesh:
        results = mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        # 唇部关键点索引（外唇轮廓）
        LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                   375, 321, 405, 314, 17, 84, 181, 91, 146]
        xs = [int(landmarks.landmark[i].x * w) for i in LIP_IDX]
        ys = [int(landmarks.landmark[i].y * h) for i in LIP_IDX]

        pad = 8
        x1 = max(min(xs) - pad, 0)
        y1 = max(min(ys) - pad, 0)
        x2 = min(max(xs) + pad, w)
        y2 = min(max(ys) + pad, h)

        return frame[y1:y2, x1:x2]