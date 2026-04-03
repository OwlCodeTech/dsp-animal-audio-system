# src/feature_extract.py
import numpy as np
import librosa

# 尝试导入配置
try:
    from src import config
except ImportError:
    import config

class FeatureAnalyzer:
    """
    频域分析与特征提取器
    职责：FFT -> STFT -> MFCC(+Delta) -> 质心/过零率 -> 保存特征
    """
    
    def __init__(self):
        self.cfg = config

    def compute_fft(self, y, sr):
        """[GUI专用] 计算 FFT 频谱 (用于画波峰图)"""
        n = len(y)
        if n == 0: return [], []
        freq = np.fft.rfftfreq(n, d=1/sr)
        mag = np.abs(np.fft.rfft(y))
        return freq, mag

    def compute_stft(self, y):
        """[GUI专用] 计算 STFT (用于画语谱图)"""
        if len(y) == 0: return np.zeros((1, 1))
        # 返回分贝刻度的频谱矩阵
        D = librosa.stft(y, n_fft=self.cfg.N_FFT, hop_length=self.cfg.HOP_LENGTH)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return D_db

    # --- [新增] 统计特征计算方法 ---
    def compute_spectral_centroid(self, y, sr):
        """计算频谱质心 (亮度)"""
        if len(y) == 0: return 0.0
        # 返回每一帧的质心
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.cfg.N_FFT, hop_length=self.cfg.HOP_LENGTH)
        # 通常 GUI 展示或者统计分析只需要一个平均值
        return np.mean(cent)

    def compute_zcr(self, y):
        """计算过零率 (噪度)"""
        if len(y) == 0: return 0.0
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=self.cfg.HOP_LENGTH)
        return np.mean(zcr)
    # ---------------------------

    def extract_39d_mfcc(self, y, sr):
        """[核心算法] 提取 39维 特征 (用于模型训练)"""
        if len(y) == 0: return np.zeros((39, 1))
        
        # 1. 静态 MFCC (13维)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.cfg.N_MFCC)
        
        # 2. 动态特征 (速度 & 加速度)
        delta1 = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 3. 堆叠 (39 x Time)
        combined = np.vstack([mfcc, delta1, delta2])
        return combined

    # --- PyQt 接口方法 ---
    def analyze_segment_memory(self, y_segment, sr):
        """
        [GUI专用] 对一个 5秒 的片段进行全套分析。
        返回字典包含：FFT, STFT, MFCC, 质心, ZCR
        """
        fft_freq, fft_mag = self.compute_fft(y_segment, sr)
        stft_db = self.compute_stft(y_segment)
        mfcc_39d = self.extract_39d_mfcc(y_segment, sr)
        
        # [新增] 调用统计特征
        centroid_val = self.compute_spectral_centroid(y_segment, sr)
        zcr_val = self.compute_zcr(y_segment)
        
        return {
            'fft_freq': fft_freq,
            'fft_mag': fft_mag,
            'stft_db': stft_db,
            'mfcc_feature': mfcc_39d,
            'centroid_mean': centroid_val, # 新增
            'zcr_mean': zcr_val            # 新增
        }

    def save_feature(self, feature_matrix, path):
        """保存特征矩阵到 .npy"""
        np.save(path, feature_matrix)