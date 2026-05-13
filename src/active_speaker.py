import cv2
import mediapipe as mp
import numpy as np

# Lip landmark indices for MAR (Mouth Aspect Ratio)
_UPPER_LIP = [82, 13, 312]
_LOWER_LIP = [87, 14, 317]
_CORNER_L, _CORNER_R = 61, 291

# Tunable thresholds
MAR_OPEN_THRESH   = 0.03   # minimum MAR to count as "open"
MAR_MOTION_THRESH = 0.010  # std(MAR history) to count as "moving"


def _calc_mar(landmarks):
    """
    Mouth Aspect Ratio: sum of vertical lip distances / (2 * horizontal span).
    Using normalized 2-D landmark coordinates so it is scale- and
    rotation-invariant within the image plane.
    """
    def pt(i):
        return np.array([landmarks[i].x, landmarks[i].y])

    vert = sum(np.linalg.norm(pt(u) - pt(l))
               for u, l in zip(_UPPER_LIP, _LOWER_LIP))
    horiz = np.linalg.norm(pt(_CORNER_L) - pt(_CORNER_R))
    return vert / (2.0 * horiz + 1e-8)


class ActiveSpeakerDetector:
    def __init__(self, max_faces=4, window=15, ema_alpha=0.35):
        self.max_faces = max_faces
        self.window = window
        self.ema_alpha = ema_alpha

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
        )
        self._histories = {i: [] for i in range(max_faces)}
        self._mar_ema   = {i: 0.0 for i in range(max_faces)}

    def _update(self, face_idx, mar):
        h = self._histories[face_idx]
        h.append(mar)
        if len(h) > self.window:
            h.pop(0)
        self._mar_ema[face_idx] = (
            self.ema_alpha * mar + (1 - self.ema_alpha) * self._mar_ema[face_idx]
        )
        variation = float(np.std(h)) if len(h) >= 3 else 0.0
        is_spk = (variation > MAR_MOTION_THRESH and
                  self._mar_ema[face_idx] > MAR_OPEN_THRESH * 0.5)
        open_conf   = float(np.clip(self._mar_ema[face_idx] / MAR_OPEN_THRESH, 0, 1))
        motion_conf = float(np.clip(variation / MAR_MOTION_THRESH, 0, 1))
        conf = 0.5 * open_conf + 0.5 * motion_conf
        return is_spk, conf

    def _process(self, frame):
        return self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def is_speaking(self, frame):
        results = self._process(frame)
        if not results.multi_face_landmarks:
            return False
        mar = _calc_mar(results.multi_face_landmarks[0].landmark)
        is_spk, _ = self._update(0, mar)
        return is_spk

    def is_speaking_with_conf(self, frame):
        results = self._process(frame)
        if not results.multi_face_landmarks:
            return False, 0.0
        mar = _calc_mar(results.multi_face_landmarks[0].landmark)
        return self._update(0, mar)

    def get_speaking_status_all(self, frame):
        """Returns [(is_speaking, conf), ...] for each detected face."""
        results = self._process(frame)
        if not results.multi_face_landmarks:
            return []
        statuses = []
        for i, face_lm in enumerate(results.multi_face_landmarks[:self.max_faces]):
            mar = _calc_mar(face_lm.landmark)
            statuses.append(self._update(i, mar))
        return statuses
