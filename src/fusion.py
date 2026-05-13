import numpy as np
from src.audio import compute_delay, delay_and_sum, mvdr_beamform, spectral_subtract

_MIC_POS = np.array([[-0.06, 0], [-0.02, 0], [0.02, 0], [0.06, 0]])


def process(audio, sr, angle, frontal_conf=1.0):
    """
    Full pipeline: spectral subtraction → DAS/MVDR soft-fusion.
    audio: (n_samples, n_mics)
    frontal_conf: 1.0 = face-on (use DAS), 0.0 = side-on (use MVDR)
    """
    n_mics = audio.shape[1]

    # Per-channel spectral subtraction before beamforming
    audio_clean = np.stack(
        [spectral_subtract(audio[:, i], sr) for i in range(n_mics)],
        axis=1,
    )

    delays   = compute_delay(_MIC_POS[:n_mics], angle)
    out_das  = delay_and_sum(audio_clean, delays, sr)
    out_mvdr = mvdr_beamform(audio_clean, sr, angle, mic_positions=_MIC_POS[:n_mics])

    return frontal_conf * out_das + (1.0 - frontal_conf) * out_mvdr
