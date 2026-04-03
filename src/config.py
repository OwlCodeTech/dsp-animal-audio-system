# src/config.py

# --- 基础音频参数 ---
TARGET_SR = 22050           # 目标采样率
TARGET_DURATION = 5.0       # 切割时长 (秒)
STRIDE = 2.5                # 滑动窗口步长 (秒)
TOP_DB = 30                 # 静音消除阈值 (分贝)

# --- 输入文件设置 ---
# 支持的音频格式 (大小写敏感，代码中已做去重处理)
FILE_EXTENSIONS = ['*.wav', '*.mp3', '*.WAV', '*.MP3']

# --- [新增] 数据集划分配置 ---
# 对应: 训练集 70%, 验证集 15%, 测试集 15%
SPLIT_NAMES = ['Train', 'Val', 'Test']
SPLIT_RATIOS = [0.7, 0.15, 0.15]

# --- 特征提取参数 ---
N_MFCC = 13                 # MFCC 维度
N_FFT = 2048                # STFT 窗口大小
HOP_LENGTH = 512            # STFT 跳步

# --- 滤波器配置 ---
# 根据物种名称自动选择滤波器
FILTER_CONFIG = {
    # --- 低频组 ---
    'elephant': {'type': 'lowpass', 'freq': 1000},
    
    # --- 水下/高频组 ---
    'dolphin': {'type': 'highpass', 'freq': 150},
    'sperm whale': {'type': 'highpass', 'freq': 150},
    
    # --- 陆地/带通组 ---
    'cat': {'type': 'bandpass', 'low': 80, 'high': 8000},
    'dog': {'type': 'bandpass', 'low': 80, 'high': 8000},
    'gallus gallus': {'type': 'bandpass', 'low': 200, 'high': 8000},
    'deer': {'type': 'bandpass', 'low': 200, 'high': 8000},
}

# 默认滤波器 (用于未知物种)
DEFAULT_FILTER = {'type': 'bandpass', 'low': 100, 'high': 8000}