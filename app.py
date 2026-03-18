import streamlit as st
import tempfile
import cv2
import numpy as np
import soundfile as sf
import os

from src.vision import VisionTracker
from src.active_speaker import ActiveSpeakerDetector
from src.audio import load_audio
from src.fusion import process

st.title("🎧 AV Beamforming Demo")
st.write("Upload a video and extract the voice of the person you are looking at.")

vision = VisionTracker()
asd = ActiveSpeakerDetector()

uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file is not None:
    # 保存视频到临时文件
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    # 读取视频第一帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        st.error("Failed to read video")
    else:
        # ⭐ Vision + ASD
        angle = vision.get_target_angle(frame)
        speaking = asd.is_speaking(frame)

        st.write(f"🎯 Angle: {angle}")
        st.write(f"🗣 Speaking: {speaking}")

        # ⚠️ 这里假设有一个音频文件（demo用）
        audio_path = "data/audio.wav"

        if os.path.exists(audio_path):
            audio, sr = load_audio(audio_path)

            if len(audio.shape) == 1:
                audio = np.stack([audio]*4, axis=1)

            if speaking and angle is not None:
                output = process(audio, sr, angle)

                out_path = "output.wav"
                sf.write(out_path, output, sr)

                st.success("✅ Processed audio:")
                st.audio(out_path)
            else:
                st.warning("No active speaker detected")
        else:
            st.warning("⚠️ Please provide data/audio.wav for demo")