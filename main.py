from src.vision import get_face_angle, get_frontal_confidence, get_lip_roi
from src.active_speaker import ActiveSpeakerDetector
from src.audio import load_audio, extract_audio_from_video, compute_delay, delay_and_sum, mvdr_beamform
from src.fusion import process

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from collections import deque

SAMPLE_RATE = 16000
BLOCK_SIZE  = 1024
N_MICS      = 1   # 笔记本通常只有1个麦克风，改成实际数量

class AVSystem:
    def __init__(self, mode="demo"):
        """
        mode="demo"  → 实时摄像头+麦克风（电脑演示用）
        mode="file"  → 读取 data/video.mp4（原有逻辑）
        """
        self.mode = mode
        self.asd  = ActiveSpeakerDetector()
        self.video_path = "data/video.mp4"

        # 实时模式：用环形缓冲区存麦克风音频
        self.audio_buffer = deque(maxlen=SAMPLE_RATE * 3)  # 存最近3秒

    # ── 实时演示模式 ──────────────────────────────────────────
    def run_demo(self):
        cap = cv2.VideoCapture(0)  # 0 = 默认摄像头

        def audio_callback(indata, frames, time, status):
            self.audio_buffer.extend(indata[:, 0])  # 存单路，演示够用

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=N_MICS,
                            blocksize=BLOCK_SIZE, callback=audio_callback):
            print("演示模式启动，按 Q 退出")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                angle        = get_face_angle(frame)
                frontal_conf = get_frontal_confidence(angle)
                speaking, _  = self.asd.is_speaking_with_conf(frame)

                # 实时取最近音频做增强
                if speaking and angle is not None and len(self.audio_buffer) > BLOCK_SIZE:
                    audio_np = np.array(self.audio_buffer)
                    audio_4ch = np.stack([audio_np] * 4, axis=1)  # 单麦模拟4路
                    output = process(audio_4ch, SAMPLE_RATE, angle, frontal_conf)
                    sd.play(output, SAMPLE_RATE)  # 实时播放增强结果

                # 在画面上显示状态
                self._draw_overlay(frame, angle, frontal_conf, speaking)
                cv2.imshow("AV Beamforming Demo", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    # ── 文件模式（原有逻辑）──────────────────────────────────
    def run_file(self):
        cap = cv2.VideoCapture(self.video_path)
        audio, sr = extract_audio_from_video(self.video_path)
        if len(audio.shape) == 1:
            audio = np.stack([audio] * 4, axis=1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            angle        = get_face_angle(frame)
            frontal_conf = get_frontal_confidence(angle)
            speaking, _  = self.asd.is_speaking_with_conf(frame)

            if speaking and angle is not None:
                output = process(audio, sr, angle, frontal_conf=frontal_conf)
                sf.write("output/output.wav", output, sr)
                print("✅ 已输出语音")

        cap.release()

    # ── 画面叠加状态信息 ─────────────────────────────────────
    def _draw_overlay(self, frame, angle, conf, speaking):
        status = "Speaking" if speaking else "Silent"
        color  = (0, 255, 0) if speaking else (0, 0, 255)
        angle_str = f"{angle:.1f}" if angle is not None else "N/A"
        cv2.putText(frame, f"Angle: {angle_str} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Frontal: {conf:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def run(self):
        if self.mode == "demo":
            self.run_demo()
        else:
            self.run_file()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    AVSystem(mode=mode).run()