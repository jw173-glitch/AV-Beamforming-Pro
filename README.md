# AV Beamforming for Target Speaker Enhancement

A multimodal audio-visual system that enhances a target speaker's voice by combining visual direction estimation with audio beamforming.

> Use vision to determine **where** to listen, and audio to determine **when** someone is speaking.

## Pipeline

```
Video → YOLO person detection → Angle estimation
      → MediaPipe FaceMesh   → Mouth open ratio → Speaking detection
             ↓
Audio → Energy-based VAD → Speech segments
      → DAS / MVDR Beamforming (soft-fused by frontal confidence)
             ↓
        Enhanced Audio Output
```

## Features

- **YOLO-based direction estimation** — detects the largest person in frame, estimates horizontal angle
- **Dual speaking detection** — visual (mouth motion) + audio (energy VAD) work in parallel
- **Adaptive beamforming** — soft-fuses Delay-and-Sum (frontal) and MVDR (off-axis) based on face angle confidence
- **Speech segment display** — Streamlit UI shows exact speaking timestamps and total duration
- **Real-time demo mode** — live webcam + microphone via `main.py`

## Setup

```bash
pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` for video audio extraction. YOLO weights (`yolov8n.pt`) are downloaded automatically on first run.

## Usage

**Web UI:**
```bash
streamlit run app.py
```
Upload a video — the app extracts audio, detects speech segments, estimates speaker angle, and outputs the beamformed audio.

**Real-time webcam demo:**
```bash
python main.py demo
```

**Process a video file (`data/video.mp4`):**
```bash
python main.py file
```
Output saved to `output/output.wav`.

## Architecture

| Module | Role |
|---|---|
| `src/vision.py` | YOLO person detection for angle; MediaPipe FaceMesh for mouth-open ratio |
| `src/active_speaker.py` | Tracks mouth motion history, returns speaking flag + confidence |
| `src/audio.py` | `detect_speech_segments` (energy VAD), `delay_and_sum`, `mvdr_beamform` (STFT domain) |
| `src/fusion.py` | Soft-fuses DAS and MVDR outputs by `frontal_conf` |
| `app.py` | Streamlit UI — audio VAD + video angle estimation + beamformed output |
| `main.py` | Real-time and file-mode runner |

## Known Limitations

- A single built-in microphone replicated across 4 channels gives no real spatial gain from beamforming. For actual spatial filtering, use a physical microphone array (e.g. ReSpeaker USB).
- YOLO detects persons, not faces — accuracy degrades if the person is partially occluded or very far from the camera.
- Mouth-motion speaking detection needs ~5 frames of warmup before producing reliable results.
