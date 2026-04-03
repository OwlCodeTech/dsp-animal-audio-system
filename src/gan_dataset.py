# src/gan_dataset.py
import torch
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset
from pathlib import Path
import os

class GANSpectrogramDataset(Dataset):
    def __init__(self, csv_path, label_map):
        self.df = pd.read_csv(csv_path)
        
        # ==========================================================
        # ✅ [核心修改] 实施白名单策略 (Report Section 1.3)
        # ==========================================================
        # 只保留有情绪维度的物种。剔除 Elephant, Dolphin, Sperm Whale
        WHITELIST = ['cat', 'dog', 'gallus gallus', 'deer']
        
        # 统一转小写进行匹配
        self.df['Species_Lower'] = self.df['Species'].str.lower().str.strip()
        
        # 强制筛选：必须是 Train 集，且必须在白名单内
        self.df = self.df[
            (self.df['Split'] == 'Train') & 
            (self.df['Species_Lower'].isin(WHITELIST))
        ].reset_index(drop=True)
        
        print(f"Dataset Filtered: Keep {len(self.df)} samples in {WHITELIST}")
        
        self.label_map = label_map
        
        # ✅ [新增] 缓存目录，解决 librosa 加载慢的问题
        self.cache_dir = Path(csv_path).parent.parent / "cache_tensors"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 路径修复 (兼容 run_pipeline 生成的相对路径)
        rel_path = self.df.iloc[idx]['AudioPath']
        # ./dataset_processed/... -> 绝对路径
        wav_path = Path(__file__).parent.parent / rel_path.replace('./', '').replace('/', os.sep)
        
        # 2. 缓存文件名
        file_id = wav_path.stem
        cache_path = self.cache_dir / f"{file_id}_gan.pt"
        
        species = self.df.iloc[idx]['Species'].lower().strip()
        label = self.label_map.get(species, 0)

        # 3. 尝试读取缓存
        if cache_path.exists():
            try:
                return torch.load(cache_path), label
            except:
                pass 

        # 4. 无缓存则计算 (librosa -> mel -> normalize)
        try:
            y, sr = librosa.load(wav_path, sr=22050, duration=5.0)
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel = librosa.power_to_db(melspec, ref=np.max)
            
            # 归一化到 [-1, 1]
            norm_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
            norm_mel = norm_mel * 2 - 1
            
            # Resize 到 128x128
            tensor_mel = torch.FloatTensor(norm_mel).unsqueeze(0)
            tensor_mel = torch.nn.functional.interpolate(
                tensor_mel.unsqueeze(0), size=(128, 128), mode='bilinear'
            ).squeeze(0)
            
            # 写入缓存
            torch.save(tensor_mel, cache_path)
            return tensor_mel, label
            
        except Exception as e:
            return torch.zeros((1, 128, 128)), label