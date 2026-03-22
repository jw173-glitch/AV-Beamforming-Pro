import streamlit as st
import tempfile
import cv2
import numpy as np
import soundfile as sf

from src.vision import get_face_angle, get_frontal_confidence
from src.audio import extract_audio_from_video, detect_speech_segments
from src.fusion import process

st.title("AV Beamforming Demo")

uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file is not None:

    # ===============================
    # 保存视频
    # ===============================
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    # ===============================
    # 提取音频
    # ===============================
    audio, sr = extract_audio_from_video(video_path)

    audio_mono = audio[:,0] if audio.ndim > 1 else audio

    # ===============================
    # 🔥 音频判断是否说话（核心）
    # ===============================
    flags, times, segments = detect_speech_segments(audio_mono, sr)

    speaking = len(segments) > 0

    # ===============================
    # 🎥 视觉：从视频中提取方向
    # ===============================
    cap = cv2.VideoCapture(video_path)

    angles = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(frame_count // 30, 1)   # 抽样30帧

    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % step != 0:
            i += 1
            continue

        # 跳过黑帧
        if frame.mean() < 10:
            i += 1
            continue

        angle = get_face_angle(frame)

        if angle is not None:
            angles.append(angle)

        i += 1

    cap.release()

    # ===============================
    # 角度统计
    # ===============================
    if len(angles) > 0:
        angle = np.median(angles)   # 更稳
    else:
        angle = None

    frontal_conf = get_frontal_confidence(angle)

    # ===============================
    # 显示结果
    # ===============================
    st.write(f"Angle: {angle:.1f} deg" if angle is not None else "Angle: N/A")
    st.write(f"Frontal confidence: {frontal_conf:.2f}")
    st.write(f"Speaking: {speaking}")

    # ===============================
    # 🔥 显示说话时间段
    # ===============================
    if segments:
        st.subheader("Speaking segments:")

        for s, e in segments:
            st.write(f"{s:.2f}s → {e:.2f}s")

        total_time = sum(e - s for s, e in segments)
        st.write(f"Total speaking time: {total_time:.2f}s")

    else:
        st.write("No speaking detected")

    # ===============================
    # Beamforming
    # ===============================
    if len(audio.shape) == 1:
        audio = np.stack([audio] * 4, axis=1)

    if speaking and angle is not None:
        output = process(audio, sr, angle)

        out_path = tempfile.mktemp(suffix=".wav")
        sf.write(out_path, output, sr)

        st.success("Processed audio:")
        st.audio(out_path)
    else:
        st.warning("No active speaker detected")