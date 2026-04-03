# src/gan_inference.py
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd
from gan_model import Generator

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cgan" / "generator_last.pth"
CSV_PATH = BASE_DIR / "metadata" / "dataset_index.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GANInference:
    def __init__(self):
        # 1. 重建全局 Map
        df = pd.read_csv(CSV_PATH)
        all_species = sorted(df['Species'].str.lower().str.strip().unique())
        self.species_map = {name: idx for idx, name in enumerate(all_species)}
        self.num_classes = len(all_species)
        
        # ✅ 白名单检查
        self.whitelist = ['cat', 'dog', 'gallus gallus', 'deer']
        
        self.generator = Generator(num_classes=self.num_classes).to(DEVICE)
        if MODEL_PATH.exists():
            self.generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.generator.eval()
            print("✅ GAN Generator loaded.")
        else:
            print("❌ Model not found.")

    def generate_audio(self, species_name, save_path):
        key = species_name.lower().strip()
        
        # ✅ 强制白名单检查
        if key not in self.whitelist:
            print(f"❌ '{key}' 未在 cGAN 中训练 (白名单: {self.whitelist})")
            return False
            
        if key not in self.species_map:
            print(f"❌ Unknown ID for {key}")
            return False
            
        idx = self.species_map[key]
        label_tensor = torch.LongTensor([idx]).to(DEVICE)
        z = torch.randn(1, 100).to(DEVICE)
        
        with torch.no_grad():
            gen_img = self.generator(z, label_tensor).squeeze().cpu().numpy()
            
        # 还原音频
        spec_db = (gen_img + 1) / 2 * 80 - 80 
        spec_power = librosa.db_to_power(spec_db)
        mel_basis = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128)
        inv_mel_basis = np.linalg.pinv(mel_basis)
        linear_spec = np.dot(inv_mel_basis, spec_power)
        y_recon = librosa.griffinlim(linear_spec, n_iter=32, hop_length=512, win_length=2048)
        
        sf.write(save_path, y_recon, 22050)
        return True
    

# === 补充：主程序入口 ===
if __name__ == "__main__":
    # 1. 实例化推理器
    gan = GANInference()
    
    # 2. 创建输出目录
    output_dir = Path("results_test")
    output_dir.mkdir(exist_ok=True)
    
    # 3. 批量测试白名单里的所有物种
    targets = ['cat', 'dog', 'deer', 'gallus gallus']
    
    print("\n🎧 开始生成测试音频...")
    for species in targets:
        save_path = output_dir / f"generated_{species.replace(' ', '_')}.wav"
        
        # 调用生成函数
        success = gan.generate_audio(species, save_path)
        
        if success:
            print(f"   [√] {species} -> {save_path}")
        else:
            print(f"   [x] {species} 生成失败")
            
    print(f"\n✅ 全部完成！请打开文件夹 {output_dir.absolute()} 试听。")