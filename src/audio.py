import numpy as np
import soundfile as sf
import subprocess
from scipy.signal import stft, istft

c = 343.0
_DEFAULT_MIC_POS = np.array([[-0.06, 0], [-0.02, 0], [0.02, 0], [0.06, 0]])


def load_audio(path):
    data, sr = sf.read(path)
    return data, sr


def extract_audio_from_video(video_path, output_wav="data/audio_sync.wav", sr=16000):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ar", str(sr), "-ac", "1", "-y", output_wav
    ], check=True)
    return load_audio(output_wav)


def compute_delay(mic_positions, angle_deg):
    """Far-field propagation delay (seconds) per mic for source at angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    direction = np.array([np.sin(angle_rad), np.cos(angle_rad)])
    return mic_positions @ direction / c


def delay_and_sum(audio, delays, sr):
    """audio: (n_samples, n_mics). Returns (n_samples,) beamformed signal."""
    n_samples, n_mics = audio.shape
    out = np.zeros(n_samples)
    for i, d in enumerate(delays):
        shift = int(round(d * sr))
        if shift >= 0:
            out[:n_samples - shift] += audio[shift:, i]
        else:
            out[-shift:] += audio[:n_samples + shift, i]
    return out / n_mics


def mvdr_beamform(audio, sr, angle_deg, mic_positions=None, nperseg=512):
    """
    MVDR beamformer in the STFT domain.
    audio: (n_samples, n_mics)
    Returns (n_samples,) enhanced signal.
    """
    if mic_positions is None:
        mic_positions = _DEFAULT_MIC_POS
    n_mics = audio.shape[1]
    delays = compute_delay(mic_positions[:n_mics], angle_deg)
    noverlap = nperseg * 3 // 4

    freqs, _, _ = stft(audio[:, 0], fs=sr, nperseg=nperseg, noverlap=noverlap)
    specs = np.array([
        stft(audio[:, i], fs=sr, nperseg=nperseg, noverlap=noverlap)[2]
        for i in range(n_mics)
    ])  # (n_mics, n_freqs, n_frames)
    n_freqs, n_frames = specs.shape[1], specs.shape[2]

    out_spec = np.zeros((n_freqs, n_frames), dtype=complex)
    reg = 1e-4 * np.eye(n_mics)

    for fi, f in enumerate(freqs):
        d = np.exp(-1j * 2 * np.pi * f * delays)
        X = specs[:, fi, :]
        R = (X @ X.conj().T) / n_frames + reg
        try:
            Rinv_d = np.linalg.solve(R, d)
        except np.linalg.LinAlgError:
            Rinv_d = np.linalg.pinv(R) @ d
        denom = d.conj() @ Rinv_d
        w = Rinv_d / (denom + 1e-12)
        out_spec[fi, :] = w.conj() @ X

    _, audio_out = istft(out_spec, fs=sr, nperseg=nperseg, noverlap=noverlap)
    audio_out = np.real(audio_out)
    n_orig = audio.shape[0]
    if len(audio_out) >= n_orig:
        return audio_out[:n_orig]
    return np.pad(audio_out, (0, n_orig - len(audio_out)))


def spectral_subtract(audio, sr, noise_frames=20, alpha=2.0, beta=0.05):
    """
    Spectral subtraction for single-channel noise reduction.
    alpha: oversubtraction factor; beta: spectral floor ratio.
    """
    if audio.ndim > 1:
        audio = audio[:, 0]
    nperseg = 512
    _, _, Z = stft(audio, fs=sr, nperseg=nperseg)
    mag, phase = np.abs(Z), np.angle(Z)
    noise_est = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - alpha * noise_est, beta * mag)
    _, audio_clean = istft(mag_clean * np.exp(1j * phase), fs=sr, nperseg=nperseg)
    audio_clean = np.real(audio_clean)
    n = len(audio)
    if len(audio_clean) >= n:
        return audio_clean[:n]
    return np.pad(audio_clean, (0, n - len(audio_clean)))


def separate_speakers(audio_path, max_speakers=4):
    """Lazy-load SepFormer to avoid heavy import at module level."""
    from speechbrain.inference.separation import SepformerSeparation
    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj04mix",
        savedir="pretrained_models/sepformer",
    )
    est_sources = separator.separate_file(path=audio_path)
    sources = est_sources.detach().cpu().numpy()
    return sources[:, :max_speakers]
