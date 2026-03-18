import numpy as np
import pyroomacoustics as pra

class Beamformer:
    def __init__(self, mic_distance=0.05, num_mics=4, fs=16000):
        self.fs = fs
        # 假设是线性阵列 (Linear Array)
        self.mic_locs = np.zeros((2, num_mics))
        self.mic_locs[0, :] = np.arange(num_mics) * mic_distance

    def process_side_speaker(self, multi_mic_signals, angle_deg):
        """使用 MVDR 算法定向增强特定角度的声音"""
        # 角度转弧度 (需补偿坐标系，假设正前方是 90°)
        phi = np.deg2rad(angle_deg + 90)
        
        # 短时傅里叶变换 (STFT)
        stft_signals = pra.transform.stft.analysis(multi_mic_signals.T, L=512, hop=256)
        
        # 初始化 MVDR 波束成形器
        bf = pra.beamforming.MVDRBeamformer(self.mic_locs, self.fs, L=512)
        bf.steer_phi = phi
        
        # 处理并合成音频
        output_stft = bf.process(stft_signals)
        return pra.transform.stft.synthesis(output_stft, L=512, hop=256)