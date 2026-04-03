# src/run_pipeline.py
import sys
import pandas as pd
import random  # [新增] 用于随机划分数据集
from pathlib import Path
from tqdm import tqdm

# --- 关键：将项目根目录加入 Python 路径 ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# 导入模块
try:
    from src.preprocess import AudioCleaner
    from src.feature_extract import FeatureAnalyzer
    from src import config
except ImportError:
    from preprocess import AudioCleaner
    from feature_extract import FeatureAnalyzer
    import config

def main():
    # 路径设置
    RAW_DIR = BASE_DIR / "dataset_raw"
    PROC_DIR = BASE_DIR / "dataset_processed"
    FEAT_DIR = BASE_DIR / "features"
    CSV_PATH = BASE_DIR / "metadata" / "dataset_index.csv"
    
    # 初始化处理器
    cleaner = AudioCleaner()
    analyzer = FeatureAnalyzer()
    
    # 确保目录存在
    for d in [PROC_DIR, FEAT_DIR, BASE_DIR / "metadata"]:
        d.mkdir(parents=True, exist_ok=True)
    
    # --- 扫描多种格式 (WAV 和 MP3) ---
    all_files = []
    if RAW_DIR.exists():
        for ext in config.FILE_EXTENSIONS:
            found = list(RAW_DIR.rglob(ext))
            all_files.extend(found)
    else:
        print(f"Error: dataset_raw directory not found at {RAW_DIR}")
        return

    # 去重并排序
    all_files = sorted(list(set(all_files)))
    
    csv_rows = []
    
    print(f"Starting pipeline for {len(all_files)} files (WAV/MP3)...")
    
    for audio_path in tqdm(all_files, desc="Processing"):
        try:
            # 1. 解析标签
            rel_path = audio_path.relative_to(RAW_DIR)
            species = rel_path.parts[0] # 一级目录为物种
            
            # 标签逻辑
            if len(rel_path.parts) > 2:
                emotion = rel_path.parts[1]
            else:
                emotion = 'angry' if species.lower() == 'elephant' else 'neutral'
            
            # 2. 调用预处理 (Process)
            # 支持 MP3/WAV 加载，返回切割后的片段列表
            segments, sr, _ = cleaner.process_single_file_memory(audio_path, species)
            
            if not segments:
                continue

            # 3. 循环处理片段
            for idx, seg in enumerate(segments):
                base_name = audio_path.stem 
                unique_name = f"{species}_{emotion}_{base_name}_{idx:03d}"
                
                # 3.1 保存预处理后的 WAV
                wav_save_path = PROC_DIR / species / f"{unique_name}.wav"
                wav_save_path.parent.mkdir(exist_ok=True)
                cleaner.save_wav(seg, wav_save_path, sr)
                
                # 4. 调用特征分析 (Analysis)
                mfcc_matrix = analyzer.extract_39d_mfcc(seg, sr)
                
                # 4.1 保存特征 .npy
                feat_save_path = FEAT_DIR / species / f"{unique_name}.npy"
                feat_save_path.parent.mkdir(exist_ok=True)
                analyzer.save_feature(mfcc_matrix, feat_save_path)
                
                # --- [新增] 数据集随机划分逻辑 ---
                # 根据 config 中的权重随机分配 Train/Val/Test
                split_group = random.choices(config.SPLIT_NAMES, weights=config.SPLIT_RATIOS, k=1)[0]
                
                # 5. 记录 CSV
                csv_rows.append({
                    "Filename": f"{unique_name}.wav",
                    "Species": species,
                    "Emotion": emotion,
                    "Duration": config.TARGET_DURATION,
                    "Split": split_group,  # [新增] 这一列用于区分数据集
                    "FeaturePath": f"./features/{species}/{unique_name}.npy",
                    "AudioPath": f"./dataset_processed/{species}/{unique_name}.wav",
                    "OriginalSource": audio_path.name
                })
                
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            continue
            
    # 保存索引
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        # 调整列顺序，让 Split 更显眼
        cols = ["Filename", "Species", "Emotion", "Split", "FeaturePath", "AudioPath", "Duration", "OriginalSource"]
        df = df[cols]
        
        df.to_csv(CSV_PATH, index=False)
        print(f"\nPipeline Finished! Processed {len(csv_rows)} segments.")
        print(f"Index saved to: {CSV_PATH}")
        
        # 打印简单的分布统计
        print("\nDataset Distribution:")
        print(df['Split'].value_counts())
    else:
        print("No files were processed successfully.")

if __name__ == "__main__":
    main()