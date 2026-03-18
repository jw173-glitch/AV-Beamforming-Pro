import numpy as np
import soundfile as sf
from scipy.signal import stft, istft  # 新增

c = 343

def load_audio(path):
    data, sr = sf.read(path)
    return data, sr

def compute_delay(mic_pos, angle):
    theta = np.deg2rad(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    delay = mic_pos @ direction / c
    return delay

def delay_and_sum(signals, delays, sr):
    aligned = []
    for i in range(signals.shape[1]):
        shift = int(delays[i] * sr)
        aligned.append(np.roll(signals[:, i], -shift))
    return np.mean(aligned, axis=0)

# ===== 新增 =====

def mvdr_beamform(signals, sr, angle):
    """
    signals: (n_samples, n_mics)
    """
    mic_pos = np.array([[-0.06,0],[-0.02,0],[0.02,0],[0.06,0]])
    n_mics = signals.shape[1]

    # STFT
    X = np.array([stft(signals[:, i], fs=sr, nperseg=512)[2] for i in range(n_mics)])
    # X: (n_mics, n_freqs, n_frames)
    _, n_freqs, n_frames = X.shape

    # 导向矢量
    theta = np.deg2rad(angle)
    freqs = np.linspace(0, sr / 2, n_freqs)
    tau = mic_pos[:, 0] * np.cos(theta) / c          # (n_mics,)
    steering = np.exp(-2j * np.pi * freqs[None, :] * tau[:, None])  # (n_mics, n_freqs)

    out = np.zeros((n_freqs, n_frames), dtype=complex)
    for f in range(n_freqs):
        Xf = X[:, f, :]                               # (n_mics, n_frames)
        R = Xf @ Xf.conj().T / n_frames
        R += np.eye(n_mics) * 1e-6                    # 正则化
        sv = steering[:, f]
        from numpy.linalg import solve
        w = solve(R, sv)
        w /= (sv.conj() @ w + 1e-8)
        out[f, :] = w.conj() @ Xf

    _, enhanced = istft(out, fs=sr, nperseg=512)
    return enhanced.real