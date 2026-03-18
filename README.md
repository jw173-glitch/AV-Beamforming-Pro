# AV Beamforming Prototype

A simple audio-visual speech enhancement system that combines:
- Vision-based direction estimation (MediaPipe FaceMesh)
- Active speaker detection (lip motion analysis)
- Beamforming (delay-and-sum)

## Pipeline

Camera → Face Detection → Angle Estimation  
+ Active Speaker Detection  
→ Beamforming → Enhanced Audio Output

## Features

- 🎯 Track target speaker based on where user is looking
- 🗣 Detect whether the target is speaking
- 🔊 Enhance target speech using beamforming

## Demo Idea

- Look at a person → system enhances their voice
- Ignore non-speaking faces

## Setup

```bash
pip install -r requirements.txt
python main.py