import numpy as np
import soundfile as sf
import subprocess
import os
from scipy.signal import stft, istft

# 声速 (m/s)
c = 343


# ===============================
# 基础功能
# ===============================

def load_audio(path):
    data, sr = sf.read(path)
    return data, sr


def extract_audio_from_video(video_path, output_path="temp_audio.wav"):
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(output_path):
        raise RuntimeError("ffmpeg failed to extract audio")

    audio, sr = sf.read(output_path)
    return audio, sr


# ===============================
# 🔥 新增：语音检测（核心）
# ===============================

def detect_speech_segments(audio, sr, frame_ms=100, threshold_ratio=1.5):
    """
    自动阈值语音检测（更稳）

    threshold_ratio:
        >1.0 越大越严格
    """

    frame_size = int(sr * frame_ms / 1000)

    energies = []
    times = []

    # ===== 计算能量 =====
    for i in range(0, len(audio), frame_size):
        segment = audio[i:i+frame_size]

        if len(segment) == 0:
            continue

        energy = np.mean(segment**2)
        energies.append(energy)
        times.append(i / sr)

    energies = np.array(energies)

    # ===== 自动阈值（关键升级🔥）=====
    noise_floor = np.percentile(energies, 20)   # 噪声水平
    threshold = noise_floor * threshold_ratio

    speaking_flags = energies > threshold

    # ===== 合并时间段 =====
    segments = []
    start = None

    for t, flag in zip(times, speaking_flags):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            segments.append((start, t))
            start = None

    if start is not None:
        segments.append((start, times[-1]))

    return speaking_flags, times, segments


# ===============================
# Beamforming 基础
# ===============================

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


# ===============================
# MVDR Beamforming（核心）
# ===============================

def mvdr_beamform(signals, sr, angle):

    mic_pos = np.array([
        [-0.06, 0],
        [-0.02, 0],
        [ 0.02, 0],
        [ 0.06, 0]
    ])

    n_mics = signals.shape[1]

    # ===== STFT =====
    X = np.array([
        stft(signals[:, i], fs=sr, nperseg=512)[2]
        for i in range(n_mics)
    ])

    _, n_freqs, n_frames = X.shape

    # ===== 导向矢量 =====
    theta = np.deg2rad(angle)
    freqs = np.linspace(0, sr / 2, n_freqs)

    tau = mic_pos[:, 0] * np.cos(theta) / c
    steering = np.exp(-2j * np.pi * freqs[None, :] * tau[:, None])

    out = np.zeros((n_freqs, n_frames), dtype=complex)

    for f in range(n_freqs):
        Xf = X[:, f, :]

        R = Xf @ Xf.conj().T / n_frames
        R += np.eye(n_mics) * 1e-6

        sv = steering[:, f]

        from numpy.linalg import solve
        w = solve(R, sv)
        w /= (sv.conj() @ w + 1e-8)

        out[f, :] = w.conj() @ Xf

    _, enhanced = istft(out, fs=sr, nperseg=512)

    return enhanced.real
# ===============================
# 🔥 语音检测（新增）
# ===============================
def detect_speech_segments(audio, sr, frame_ms=100, threshold_ratio=1.5):

    frame_size = int(sr * frame_ms / 1000)

    energies = []
    times = []

    for i in range(0, len(audio), frame_size):
        segment = audio[i:i+frame_size]

        if len(segment) == 0:
            continue

        energy = np.mean(segment**2)
        energies.append(energy)
        times.append(i / sr)

    energies = np.array(energies)

    # 自动阈值
    noise_floor = np.percentile(energies, 20)
    threshold = noise_floor * threshold_ratio

    speaking_flags = energies > threshold

    # 合并时间段
    segments = []
    start = None

    for t, flag in zip(times, speaking_flags):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            segments.append((start, t))
            start = None

    if start is not None:
        segments.append((start, times[-1]))

    return speaking_flags, times, segments