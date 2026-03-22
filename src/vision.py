import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ===============================
# YOLO（检测人）
# ===============================
model = YOLO("yolov8n.pt")

# ===============================
# FaceMesh（检测嘴）
# ===============================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.3
)

# ===============================
# 角度（YOLO）
# ===============================
def get_face_angle(frame, fov=90):
    h, w = frame.shape[:2]

    results = model(frame, verbose=False)

    if len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()

    # 选最大目标
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = np.argmax(areas)

    x1, y1, x2, y2 = boxes[idx]

    # 用上1/4作为头部近似
    cx = ((x1 + x2) / 2) / w

    angle = (cx - 0.5) * fov

    return angle


# ===============================
# 正面置信度
# ===============================
def get_frontal_confidence(angle, max_yaw=45.0):
    if angle is None:
        return 0.0

    return float(np.clip(1.0 - abs(angle) / max_yaw, 0.0, 1.0))


# ===============================
# 嘴部开合（关键）
# ===============================
def get_mouth_open_ratio(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]

    # 上下嘴唇点
    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]

    return abs(top.y - bottom.y)