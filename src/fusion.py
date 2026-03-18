import numpy as np
from src.audio import compute_delay, delay_and_sum, mvdr_beamform  # 新增 mvdr_beamform

def process(audio, sr, angle, frontal_conf=1.0):  # 新增 frontal_conf 参数
    mic_pos = np.array([
        [-0.06, 0],
        [-0.02, 0],
        [0.02, 0],
        [0.06, 0]
    ])

    delays = compute_delay(mic_pos, angle)
    out_das = delay_and_sum(audio, delays, sr)      # 原有路径

    # 新增：侧面时用 MVDR，软融合
    out_mvdr = mvdr_beamform(audio, sr, angle)
    output = frontal_conf * out_das + (1.0 - frontal_conf) * out_mvdr

    return output