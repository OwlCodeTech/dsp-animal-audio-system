# src/generate_stats.py
import sys
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# 路径定位
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from src.feature_extract import FeatureAnalyzer
    from src import config
except ImportError:
    from feature_extract import FeatureAnalyzer
    import config

def main():
    # 1. 设置路径
    # 注意：这里我们去读取【预处理后】的 wav 文件，因为它们已经去噪且长度统一
    PROC_DIR = BASE_DIR / "dataset_processed"
    STATS_CSV_PATH = BASE_DIR / "metadata" / "species_statistics.csv"
    
    if not PROC_DIR.exists():
        print(f"Error: 找不到预处理数据 {PROC_DIR}，请先运行 run_pipeline.py")
        return

    # 2. 初始化分析器
    analyzer = FeatureAnalyzer()
    
    # 3. 扫描所有 wav 文件
    all_wavs = list(PROC_DIR.rglob("*.wav"))
    stats_rows = []
    
    print(f"正在计算统计特征 (用于绘制箱线图)... 总文件数: {len(all_wavs)}")
    
    for wav_path in tqdm(all_wavs, desc="Analyzing Stats"):
        try:
            # 加载音频 (22050Hz)
            y, sr = librosa.load(wav_path, sr=config.TARGET_SR)
            
            # 解析物种 (父文件夹名)
            # 例如: dataset_processed/cat/cat_alarm_001.wav -> cat
            species = wav_path.parent.name
            
            # [核心逻辑] 调用 feature_extract 中的新方法
            centroid = analyzer.compute_spectral_centroid(y, sr)
            zcr = analyzer.compute_zcr(y)
            
            stats_rows.append({
                "Filename": wav_path.name,
                "Species": species,
                "Centroid_Mean": centroid,
                "ZCR_Mean": zcr
            })
            
        except Exception as e:
            print(f"Error analyzing {wav_path.name}: {e}")
            continue
            
    # 4. 保存结果
    if stats_rows:
        df = pd.DataFrame(stats_rows)
        df.to_csv(STATS_CSV_PATH, index=False)
        print(f"\n统计完成！数据已保存至: {STATS_CSV_PATH}")
        print("您现在可以使用此 CSV 在论文中绘制箱线图了。")
        
        # 简单打印均值预览
        print("\n--- 各物种特征均值预览 ---")
        print(df.groupby("Species")[["Centroid_Mean", "ZCR_Mean"]].mean())
    else:
        print("未生成任何数据。")

if __name__ == "__main__":
    main()