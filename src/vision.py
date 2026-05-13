import cv2
import mediapipe as mp
import numpy as np

mp_face      = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
           375, 321, 405, 314, 17, 84, 181, 91, 146]


class VisionTracker:
    """
    Persistent MediaPipe instances for efficient per-frame inference.
    Use as a context manager or call close() when done.
    """

    def __init__(self, fov=90, max_faces=4):
        self.fov = fov
        self.max_faces = max_faces
        self._detector = mp_face.FaceDetection(model_selection=0)
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=max_faces,
        )

    def close(self):
        self._detector.close()
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _rgb(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_angle(self, frame):
        """
        Returns (angle_deg, frontal_conf) for the primary detected face.
        Refines bbox-center angle using nose/eye asymmetry from FaceMesh.
        """
        rgb = self._rgb(frame)
        det_res  = self._detector.process(rgb)
        mesh_res = self._mesh.process(rgb)

        if not det_res.detections:
            return None, 0.0

        bbox = det_res.detections[0].location_data.relative_bounding_box
        cx = bbox.xmin + bbox.width / 2

        if mesh_res.multi_face_landmarks:
            lm = mesh_res.multi_face_landmarks[0].landmark
            # Use nose-tip offset from eye midpoint as yaw proxy
            eye_mid_x = (lm[33].x + lm[263].x) / 2
            eye_span  = abs(lm[263].x - lm[33].x) + 1e-8
            yaw_norm  = (lm[1].x - eye_mid_x) / eye_span
            cx += 0.15 * yaw_norm  # small refinement

        angle = (cx - 0.5) * self.fov
        frontal_conf = float(np.clip(1.0 - abs(angle) / 45.0, 0.0, 1.0))
        return angle, frontal_conf

    def get_all_angles(self, frame):
        """Returns list of face angles (degrees) sorted left to right."""
        det_res = self._detector.process(self._rgb(frame))
        if not det_res.detections:
            return []
        angles = []
        for det in det_res.detections[:self.max_faces]:
            bbox = det.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2
            angles.append((cx - 0.5) * self.fov)
        return sorted(angles)


# ── Stateless helpers kept for backward compatibility ──────────────────────

def get_face_angle(frame, fov=90):
    h, w, _ = frame.shape
    with mp_face.FaceDetection(model_selection=0) as detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2
            return (cx - 0.5) * fov
    return None


def get_face_angles(frame, fov=90, max_faces=4):
    with mp_face.FaceDetection(model_selection=0) as detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return []
        angles = []
        for det in results.detections[:max_faces]:
            bbox = det.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2
            angles.append((cx - 0.5) * fov)
        return sorted(angles)


def get_frontal_confidence(angle, max_yaw=45.0):
    if angle is None:
        return 0.0
    return float(np.clip(1.0 - abs(angle) / max_yaw, 0.0, 1.0))


def get_lip_roi(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(
        static_image_mode=False, refine_landmarks=True, max_num_faces=1
    ) as mesh:
        results = mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        xs = [int(landmarks.landmark[i].x * w) for i in LIP_IDX]
        ys = [int(landmarks.landmark[i].y * h) for i in LIP_IDX]
        pad = 8
        return frame[max(min(ys)-pad, 0):min(max(ys)+pad, h),
                     max(min(xs)-pad, 0):min(max(xs)+pad, w)]


def get_all_lip_rois(frame, max_faces=4):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_list = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=False, refine_landmarks=True, max_num_faces=max_faces
    ) as mesh:
        results = mesh.process(rgb)
        if not results.multi_face_landmarks:
            return []
        with mp_face.FaceDetection(model_selection=0) as detector:
            det_results = detector.process(rgb)
            detections = det_results.detections[:max_faces] if det_results.detections else []
        for i, landmarks in enumerate(results.multi_face_landmarks[:max_faces]):
            xs = [int(landmarks.landmark[j].x * w) for j in LIP_IDX]
            ys = [int(landmarks.landmark[j].y * h) for j in LIP_IDX]
            pad = 8
            roi = frame[max(min(ys)-pad, 0):min(max(ys)+pad, h),
                        max(min(xs)-pad, 0):min(max(xs)+pad, w)]
            angle = None
            if i < len(detections):
                bbox = detections[i].location_data.relative_bounding_box
                cx = bbox.xmin + bbox.width / 2
                angle = (cx - 0.5) * 90
            results_list.append((angle, roi))
    return sorted(results_list, key=lambda x: x[0] if x[0] is not None else 0)
