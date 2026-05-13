import streamlit as st
import tempfile
import cv2
import numpy as np
import soundfile as sf

from src.vision import get_face_angle, get_frontal_confidence
from src.active_speaker import ActiveSpeakerDetector
from src.audio import load_audio, extract_audio_from_video, spectral_subtract
from src.fusion import process

N_SAMPLE_FRAMES = 12   # frames to sample for angle/speaking aggregation
SPEAKING_RATIO   = 0.3  # fraction of frames that must show speaking


def sample_frames(video_path, n=N_SAMPLE_FRAMES):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        ret, frame = cap.read()
        cap.release()
        return [frame] if ret else []
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


st.set_page_config(page_title="AV Beamforming", layout="wide")
st.title("AV Beamforming Demo")
st.write("Upload a video — the system enhances the voice of the detected speaker.")

with st.sidebar:
    st.header("Settings")
    fov       = st.slider("Camera FoV (deg)", 60, 120, 90)
    max_yaw   = st.slider("Frontal max yaw (deg)", 20, 60, 45)
    ss_alpha  = st.slider("Noise reduction strength", 1.0, 4.0, 2.0, 0.5)

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    with st.spinner("Extracting audio..."):
        try:
            audio, sr = extract_audio_from_video(video_path)
        except Exception as e:
            st.error(f"ffmpeg failed: {e}")
            st.stop()

    with st.spinner(f"Analysing {N_SAMPLE_FRAMES} frames..."):
        frames = sample_frames(video_path)
        if not frames:
            st.error("Could not read video frames.")
            st.stop()

        asd    = ActiveSpeakerDetector()
        angles, confs, speaking_votes = [], [], []

        for frame in frames:
            angle = get_face_angle(frame, fov=fov)
            if angle is not None:
                fc = get_frontal_confidence(angle, max_yaw=max_yaw)
                angles.append(angle)
                confs.append(fc)
            spk, _ = asd.is_speaking_with_conf(frame)
            speaking_votes.append(spk)

    col1, col2, col3 = st.columns(3)

    if not angles:
        st.warning("No face detected in the video.")
        st.stop()

    final_angle        = float(np.median(angles))
    final_frontal_conf = float(np.median(confs))
    speaking_ratio     = sum(speaking_votes) / len(speaking_votes)
    final_speaking     = speaking_ratio >= SPEAKING_RATIO

    col1.metric("Angle", f"{final_angle:.1f}°")
    col2.metric("Frontal confidence", f"{final_frontal_conf:.2f}")
    col3.metric("Speaking frames", f"{speaking_ratio*100:.0f}%",
                delta="active" if final_speaking else "silent")

    # Always show original audio for comparison
    raw_path = tempfile.mktemp(suffix="_raw.wav")
    sf.write(raw_path, audio, sr)
    st.subheader("Original audio")
    st.audio(raw_path)

    if final_speaking and final_angle is not None:
        with st.spinner("Beamforming + noise reduction..."):
            if audio.ndim == 1:
                audio_4ch = np.stack([audio] * 4, axis=1)
            else:
                audio_4ch = np.stack([audio[:, 0]] * 4, axis=1)

            output = process(audio_4ch, sr, final_angle, frontal_conf=final_frontal_conf)

        out_path = tempfile.mktemp(suffix="_enhanced.wav")
        sf.write(out_path, output, sr)
        st.subheader("Enhanced audio (beamformed + noise reduced)")
        st.audio(out_path)
        st.success(f"Processed — angle {final_angle:.1f}°, "
                   f"frontal conf {final_frontal_conf:.2f}")
    else:
        st.warning("No active speaker detected — showing noise-reduced audio only.")
        with st.spinner("Applying noise reduction..."):
            audio_clean = spectral_subtract(audio, sr)
        clean_path = tempfile.mktemp(suffix="_clean.wav")
        sf.write(clean_path, audio_clean, sr)
        st.subheader("Noise-reduced audio")
        st.audio(clean_path)
