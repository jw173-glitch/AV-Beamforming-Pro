from collections import deque
import numpy as np
from src.vision import get_mouth_open_ratio

class ActiveSpeakerDetector:
    def __init__(self, window_size=5, threshold=0.01):
        self.history = deque(maxlen=window_size)
        self.threshold = threshold

    def is_speaking_with_conf(self, frame):
        ratio = get_mouth_open_ratio(frame)

        if ratio is None:
            return False, 0.0

        self.history.append(ratio)

        if len(self.history) < 2:
            return False, 0.0

        # 计算变化（关键）
        diffs = np.abs(np.diff(self.history))
        motion = np.mean(diffs)

        speaking = motion > self.threshold

        return speaking, motion