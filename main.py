import cv2
import numpy as np
from src.vision import VisionTracker
from src.audio import Beamformer

def main():
    tracker = VisionTracker()
    # 这里的 mic_distance 后续可以根据 Gao 提供的参数调整
    bf = Beamformer(mic_distance=0.05, num_mics=4)
    
    # 模拟从摄像头读取一帧
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8) # 模拟帧
    mock_audio = np.random.randn(4, 16000).astype(np.float32) # 模拟 4 路音频
    
    angle = tracker.get_target_angle(mock_frame)
    
    if angle is not None:
        if abs(angle) <= 30:
            print(f">>> 状态: 正前方 ({angle:.1f}°)")
            print(">>> 策略: 激活【视听融合 AI】模式，调用知识蒸馏后的轻量化模型。")
            # 这里的 TODO: 接入 SpeechBrain 模型
        else:
            print(f">>> 状态: 侧面 ({angle:.1f}°)")
            print(">>> 策略: 激活【物理波束成形】模式，压制正向背景音。")
            enhanced_audio = bf.process_side_speaker(mock_audio, angle)
    else:
        print(">>> 状态: 未检测到目标。保持原始增益。")

if __name__ == "__main__":
    main()