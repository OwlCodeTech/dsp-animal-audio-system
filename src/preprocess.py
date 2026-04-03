# src/preprocess.py
import numpy as np
import librosa
import scipy.signal as signal
import soundfile as sf
from pathlib import Path

# 尝试导入配置
try:
    from src import config
except ImportError:
    import config

class AudioCleaner:
    """
    音频清洗与预处理器
    职责：加载(WAV/MP3) -> 静音消除 -> 滤波 -> 归一化 -> 切割
    """
    
    def __init__(self):
        self.cfg = config

    def apply_filter(self, y, sr, species_name):
        """[核心算法] 应用巴特沃斯滤波器"""
        species_key = species_name.lower()
        conf = self.cfg.FILTER_CONFIG.get(species_key, self.cfg.DEFAULT_FILTER)
        
        order = 4
        nyquist = 0.5 * sr
        
        try:
            if conf['type'] == 'lowpass':
                b, a = signal.butter(order, conf['freq'] / nyquist, btype='low')
            elif conf['type'] == 'highpass':
                b, a = signal.butter(order, conf['freq'] / nyquist, btype='high')
            elif conf['type'] == 'bandpass':
                low = conf['low'] / nyquist
                high = conf['high'] / nyquist
                b, a = signal.butter(order, [low, high], btype='band')
            
            return signal.filtfilt(b, a, y)
        except Exception as e:
            print(f"Filter Warning ({species_name}): {e}")
            return y

    def normalize(self, y):
        """[核心算法] 能量归一化"""
        max_val = np.max(np.abs(y))
        return y / max_val if max_val > 0 else y

    def segment_signal(self, y, sr):
        """[核心算法] 滑动窗口切割 + 循环填充"""
        target_len = int(self.cfg.TARGET_DURATION * sr)
        stride = int(self.cfg.STRIDE * sr)
        segments = []
        
        # 1. 音频过短 -> 循环填充
        if len(y) < target_len:
            padding = np.tile(y, int(np.ceil(target_len / len(y))))
            return [padding[:target_len]]
        
        # 2. 音频够长 -> 滑动切割
        for i in range(0, len(y), stride):
            chunk = y[i : i + target_len]
            # 处理尾部不足的情况
            if len(chunk) < target_len:
                if len(chunk) > target_len * 0.1: # 只保留长度超过10%的尾部
                    padding = np.tile(chunk, int(np.ceil(target_len / len(chunk))))
                    segments.append(padding[:target_len])
            else:
                segments.append(chunk)
                
        return segments

    # --- PyQt / Pipeline 接口方法 ---
    def process_single_file_memory(self, file_path, species="unknown"):
        """
        给定文件路径(支持wav/mp3)，返回清洗后且切割好的音频片段列表。
        """
        # 1. 加载
        try:
            # librosa.load 内部调用 ffmpeg，支持 mp3
            y, sr = librosa.load(file_path, sr=self.cfg.TARGET_SR, mono=True)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return [], self.cfg.TARGET_SR, np.array([])
        
        # 2. 清洗
        y_trim, _ = librosa.effects.trim(y, top_db=self.cfg.TOP_DB)
        y_filt = self.apply_filter(y_trim, sr, species)
        y_norm = self.normalize(y_filt)
        
        # 3. 切割
        segments = self.segment_signal(y_norm, sr)
        
        return segments, sr, y_trim

    def save_wav(self, y, path, sr):
        """保存音频到磁盘，强制保存为 WAV 格式"""
        sf.write(path, y, sr, subtype='PCM_16')