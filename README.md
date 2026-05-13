# AV Beamforming Pro

Audio-visual speech enhancement system. Detects who is speaking from camera input and enhances their voice using beamforming + noise reduction.

## Pipeline

```
Camera → Face Detection (MediaPipe) → Angle Estimation
       → Active Speaker Detection (MAR-based lip analysis)
              ↓
Audio → Spectral Subtraction → DAS / MVDR Beamforming (soft-fused)
              ↓
         Enhanced Audio Output
```

## Features

- **Multi-face tracking** — detects up to 4 faces simultaneously
- **MAR-based speaking detection** — Mouth Aspect Ratio with EMA smoothing; robust to head movement and lighting changes
- **Adaptive beamforming** — MVDR (STFT domain) for side angles, Delay-and-Sum for frontal; soft-fused by confidence
- **Spectral subtraction** — per-channel noise reduction before beamforming
- **Streamlit web UI** — upload a video, compare original vs. enhanced audio side by side
- **Real-time demo mode** — live webcam + microphone via `main.py`

## Setup

```bash
pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` for video audio extraction.

## Usage

**Web UI (recommended for demo):**
```bash
streamlit run app.py
```

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
| `src/vision.py` | Face detection + angle estimation. `VisionTracker` keeps persistent MediaPipe instances for real-time use. |
| `src/active_speaker.py` | Computes MAR per face per frame, keeps a 15-frame history with EMA smoothing, combines opening + motion confidence. |
| `src/audio.py` | `compute_delay`, `delay_and_sum`, `mvdr_beamform` (STFT domain), `spectral_subtract`. SepFormer separation is lazy-loaded. |
| `src/fusion.py` | Orchestrates spectral subtraction → beamforming → soft DAS/MVDR fusion. |
| `main.py` | Real-time demo and file-mode runner. |
| `app.py` | Streamlit UI — samples 12 frames for stable angle/speaking detection, shows original vs. enhanced audio. |

## Limitations

- A laptop's single built-in microphone is replicated across 4 virtual channels; beamforming provides no spatial gain in this case. For real spatial filtering, connect a physical USB microphone array (e.g. ReSpeaker).
- The speaking detector needs ~10 frames of context to warm up; the first few seconds may be unreliable.
- MVDR STFT processing is CPU-heavy; real-time use may need GPU or reduced `nperseg`.
