import streamlit as st
import tempfile
import cv2
import numpy as np
import soundfile as sf
import os

from src.vision import get_face_angle, get_frontal_confidence   # 改
from src.active_speaker import ActiveSpeakerDetector
from src.audio import load_audio, extract_audio_from_video      # 改
from src.fusion import process

st.title("AV Beamforming Demo")
st.write("Upload a video and extract the voice of the person you are looking at.")

asd = ActiveSpeakerDetector()  # 删掉 VisionTracker

uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    # 改：从视频提取音频，天然同步，不再依赖 data/audio.wav
    with st.spinner("Extracting audio from video..."):
        try:
            audio, sr = extract_audio_from_video(video_path)
        except Exception as e:
            st.error(f"ffmpeg 提取音频失败：{e}")
            st.stop()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to read video")
        st.stop()

    angle        = get_face_angle(frame)                # 改
    frontal_conf = get_frontal_confidence(angle)        # 新增
    speaking, _  = asd.is_speaking_with_conf(frame)    # 改：带置信度版本

    st.write(f"Angle: {angle:.1f} deg" if angle is not None else "Angle: N/A")
    st.write(f"Frontal confidence: {frontal_conf:.2f}")
    st.write(f"Speaking: {speaking}")

    if len(audio.shape) == 1:
        audio = np.stack([audio] * 4, axis=1)

    if speaking and angle is not None:
        # 改：传入 frontal_conf 启用软融合
        output = process(audio, sr, angle, frontal_conf=frontal_conf)
        out_path = tempfile.mktemp(suffix=".wav")
        sf.write(out_path, output, sr)
        st.success("Processed audio:")
        st.audio(out_path)
    else:
        st.warning("No active speaker detected")