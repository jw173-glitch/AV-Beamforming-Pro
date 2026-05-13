from src.vision import VisionTracker, get_face_angle, get_frontal_confidence
from src.active_speaker import ActiveSpeakerDetector
from src.audio import load_audio, extract_audio_from_video
from src.fusion import process

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from collections import deque

SAMPLE_RATE = 16000
BLOCK_SIZE  = 1024
N_MICS      = 1   # set to actual mic count


class AVSystem:
    def __init__(self, mode="demo"):
        """
        mode="demo"  → real-time webcam + mic
        mode="file"  → read data/video.mp4
        """
        self.mode       = mode
        self.asd        = ActiveSpeakerDetector()
        self.video_path = "data/video.mp4"
        self.audio_buffer = deque(maxlen=SAMPLE_RATE * 3)

    # ── Real-time demo ────────────────────────────────────────────────────
    def run_demo(self):
        cap = cv2.VideoCapture(0)

        def audio_callback(indata, frames, time, status):
            self.audio_buffer.extend(indata[:, 0])

        with VisionTracker() as tracker, \
             sd.InputStream(samplerate=SAMPLE_RATE, channels=N_MICS,
                            blocksize=BLOCK_SIZE, callback=audio_callback):
            print("Demo mode started — press Q to quit")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                angle, frontal_conf = tracker.get_angle(frame)
                speaking, spk_conf  = self.asd.is_speaking_with_conf(frame)

                if speaking and angle is not None and len(self.audio_buffer) > BLOCK_SIZE:
                    audio_np  = np.array(self.audio_buffer)
                    audio_4ch = np.stack([audio_np] * 4, axis=1)
                    output    = process(audio_4ch, SAMPLE_RATE, angle, frontal_conf)
                    sd.play(output, SAMPLE_RATE)

                self._draw_overlay(frame, angle, frontal_conf, speaking, spk_conf)
                cv2.imshow("AV Beamforming Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    # ── File mode ─────────────────────────────────────────────────────────
    def run_file(self):
        cap        = cv2.VideoCapture(self.video_path)
        audio, sr  = extract_audio_from_video(self.video_path)
        if audio.ndim == 1:
            audio = np.stack([audio] * 4, axis=1)

        with VisionTracker() as tracker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                angle, frontal_conf = tracker.get_angle(frame)
                speaking, _         = self.asd.is_speaking_with_conf(frame)

                if speaking and angle is not None:
                    output = process(audio, sr, angle, frontal_conf=frontal_conf)
                    sf.write("output/output.wav", output, sr)
                    print("Saved enhanced audio → output/output.wav")

        cap.release()

    # ── Overlay ───────────────────────────────────────────────────────────
    def _draw_overlay(self, frame, angle, conf, speaking, spk_conf):
        status    = f"Speaking ({spk_conf:.2f})" if speaking else "Silent"
        color     = (0, 255, 0) if speaking else (0, 0, 255)
        angle_str = f"{angle:.1f}" if angle is not None else "N/A"
        cv2.putText(frame, f"Angle: {angle_str} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frontal: {conf:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
