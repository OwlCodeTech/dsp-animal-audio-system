# src/train_cgan.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import shutil
import random
import pandas as pd

from gan_model import Generator, Discriminator
from gan_dataset import GANSpectrogramDataset
from train_species import SpeciesCNN

# 尝试导入 FID 计算库
try:
    from torch_fidelity import calculate_metrics
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("⚠️ 未检测到 torch-fidelity，FID/IS 宏观评估功能将跳过。")

# ================= 配置区域 =================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "metadata" / "dataset_index.csv"
MODEL_SAVE_DIR = BASE_DIR / "models" / "cgan"
REPORT_DIR = BASE_DIR / "report_assets" / "cgan_training"
RESUME_PATH = MODEL_SAVE_DIR / "resume.pth"
CLASSIFIER_PATH = BASE_DIR / "models" / "species_classifier_best.pth"

# 确保目录存在
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# === 训练超参数 ===
EPOCHS = 200
BATCH_SIZE = 32
LATENT_DIM = 100
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TTUR 策略
LR_G = 0.0004
LR_D = 0.0001
BETA1 = 0.5 

N_CRITIC = 1
N_GENERATOR = 3
MACRO_EVAL_INTERVAL = 20

# ================= 辅助功能函数 =================

def add_instance_noise(images, current_epoch, max_epochs=50, initial_std=0.1):
    """实例噪声：在前 max_epochs 内线性衰减"""
    if current_epoch >= max_epochs:
        return images
    factor = 1.0 - (current_epoch / max_epochs)
    std = initial_std * factor
    noise = torch.randn_like(images) * std
    return images + noise

def setup_monitor_dirs(species_map):
    """创建分物种监控目录结构 (保留原逻辑)"""
    monitor_paths = {}
    
    # 00. 宏观目录
    overall_dir = REPORT_DIR / "00_OVERALL_METRICS"
    overall_dir.mkdir(exist_ok=True)
    monitor_paths['overall'] = overall_dir
    
    # 分物种目录
    for species in species_map.keys():
        s_dir = REPORT_DIR / species
        (s_dir / "samples_fixed").mkdir(parents=True, exist_ok=True)
        (s_dir / "samples_random").mkdir(parents=True, exist_ok=True)
        monitor_paths[species] = s_dir
        
    return monitor_paths

def update_metrics_plot(history, save_path, title):
    """绘制双轴指标图 (保留原逻辑)"""
    if len(history['conf']) < 1: return
    epochs = range(len(history['conf']))
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Semantic Confidence', color=color)
    ax1.plot(epochs, history['conf'], color=color, label='Confidence')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('D-Score (Realism)', color=color)
    ax2.plot(epochs, history['d_score'], color=color, linestyle='--', label='D-Score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ================= 核心监控逻辑 =================

class TrainingMonitor:
    def __init__(self, generator, discriminator, supervisor, species_map, fixed_noise_dict):
        self.g = generator
        self.d = discriminator
        self.s = supervisor
        self.species_map = species_map
        self.paths = setup_monitor_dirs(species_map)
        self.fixed_noise = fixed_noise_dict
        
        self.micro_history = {sp: {'conf': [], 'd_score': []} for sp in species_map}
        self.macro_history = {'fid': [], 'is': [], 'epoch': []}

    def micro_eval(self, epoch):
        """微观监控：每 Epoch 执行"""
        self.g.eval()
        with torch.no_grad():
            for species, label_idx in self.species_map.items():
                # 1. 生成固定噪声样本
                z_fixed = self.fixed_noise[species]
                labels = torch.LongTensor([label_idx] * z_fixed.size(0)).to(DEVICE)
                gen_fixed = self.g(z_fixed, labels)
                
                save_image(gen_fixed, self.paths[species] / "samples_fixed" / f"epoch_{epoch:03d}.png", nrow=4, normalize=True)
                
                # 2. 生成随机样本用于打分
                n_eval = 100
                z_rand = torch.randn(n_eval, LATENT_DIM).to(DEVICE)
                labels_rand = torch.LongTensor([label_idx] * n_eval).to(DEVICE)
                gen_rand = self.g(z_rand, labels_rand) 

                if epoch % 10 == 0:
                    save_image(gen_rand[:16], self.paths[species] / "samples_random" / f"epoch_{epoch:03d}.png", nrow=4, normalize=True)

                # 3. 计算语义置信度 (✅ 新增桥接逻辑)
                # GAN(-1,1) -> (0,1) -> Resize(39,216) -> Classifier
                gen_bridge = (gen_rand + 1) * 0.5 
                gen_resized = torch.nn.functional.interpolate(gen_bridge, size=(39, 216), mode='bilinear')
                
                cls_logits = self.s(gen_resized)
                cls_probs = torch.softmax(cls_logits, dim=1)
                avg_conf = cls_probs[:, label_idx].mean().item()
                
                # 4. 计算 D-Score
                d_out = self.d(gen_rand, labels_rand)
                avg_d_score = d_out.mean().item()
                
                # 记录与绘图
                self.micro_history[species]['conf'].append(avg_conf)
                self.micro_history[species]['d_score'].append(avg_d_score)
                update_metrics_plot(self.micro_history[species], self.paths[species] / "specific_metrics.png", f"{species} Metrics")
        self.g.train()

    def macro_eval(self, epoch, real_dataset_path):
        """宏观监控：FID/IS 计算 (保留原逻辑)"""
        if not HAS_FID: return
        print(f"🌟 [Macro] Epoch {epoch} Calculating FID...")
        fake_path = REPORT_DIR / "temp_fid_fake"
        fake_path.mkdir(exist_ok=True)
        
        self.g.eval()
        n_samples = 1000
        batch_sz = 32
        done = 0
        with torch.no_grad():
            while done < n_samples:
                z = torch.randn(batch_sz, LATENT_DIM).to(DEVICE)
                rand_labels = torch.tensor(np.random.choice(list(self.species_map.values()), batch_sz), dtype=torch.long).to(DEVICE)
                gen_imgs = self.g(z, rand_labels)
                for j in range(len(gen_imgs)):
                    if done < n_samples:
                        save_image(gen_imgs[j], fake_path / f"{done}.png", normalize=True)
                        done += 1
        
        try:
            metrics = calculate_metrics(input1=str(fake_path), input2=str(real_dataset_path), cuda=True, isc=True, fid=True, verbose=False)
            fid = metrics['frechet_inception_distance']
            self.macro_history['epoch'].append(epoch)
            self.macro_history['fid'].append(fid)
            print(f"📊 FID: {fid:.2f}")
            
            plt.figure()
            plt.plot(self.macro_history['epoch'], self.macro_history['fid'])
            plt.title('FID Curve')
            plt.savefig(self.paths['overall'] / "fid_curve.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ FID Error: {e}")
        
        try: shutil.rmtree(fake_path)
        except: pass
        self.g.train()

# ================= 主训练逻辑 =================

def train():
    # 1. 准备全局映射 (✅ 关键修复)
    # 必须扫描 CSV 中所有物种，以匹配预训练分类器的 7 类权重
    df = pd.read_csv(CSV_PATH)
    all_species = sorted(df['Species'].str.lower().str.strip().unique())
    label_map = {name: idx for idx, name in enumerate(all_species)}
    num_classes = len(all_species)
    print(f"🌍 Global Label Map ({num_classes} classes): {label_map}")

    # 2. 初始化模型
    generator = Generator(num_classes=num_classes, latent_dim=LATENT_DIM).to(DEVICE)
    discriminator = Discriminator(num_classes=num_classes).to(DEVICE)
    
    # 加载 Supervisor
    supervisor = SpeciesCNN(num_classes).to(DEVICE)
    if CLASSIFIER_PATH.exists():
        supervisor.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        print("✅ Supervisor Loaded.")
    else:
        print("❌ Warning: Pretrained classifier not found!")
    supervisor.eval()
    
    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    
    adversarial_loss = nn.MSELoss()
    # ✅ [关键修改] 强语义引导: CrossEntropy
    semantic_criterion = nn.CrossEntropyLoss()

    # 3. 初始化监控器 (只监控白名单)
    WHITELIST = ['cat', 'dog', 'gallus gallus', 'deer']
    monitor_map = {k: v for k, v in label_map.items() if k in WHITELIST}
    fixed_noise_dict = {sp: torch.randn(16, LATENT_DIM).to(DEVICE) for sp in monitor_map}
    
    # FID 缓存逻辑
    real_img_cache = REPORT_DIR / "real_images_cache"
    if HAS_FID and not real_img_cache.exists():
        print("📦 Caching real images for FID...")
        real_img_cache.mkdir(parents=True, exist_ok=True)
        temp_ds = GANSpectrogramDataset(CSV_PATH, label_map)
        temp_dl = DataLoader(temp_ds, batch_size=1, shuffle=True)
        for idx, (img, _) in enumerate(temp_dl):
            if idx >= 1000: break
            save_image(img, real_img_cache / f"{idx}.png", normalize=True)
    
    monitor = TrainingMonitor(generator, discriminator, supervisor, monitor_map, fixed_noise_dict)

    # 4. 数据加载 (✅ Dataset 内部已做白名单过滤)
    dataset = GANSpectrogramDataset(CSV_PATH, label_map)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    print(f"🚀 Training Started... Monitoring: {list(monitor_map.keys())}")
    
    g_losses, d_losses = [], []

    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            real_imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            valid = torch.full((batch_size, 1), 0.9).to(DEVICE)
            fake = torch.full((batch_size, 1), 0.0).to(DEVICE)

            # --- D ---
            optimizer_D.zero_grad()
            noisy_real = add_instance_noise(real_imgs, epoch)
            d_real_loss = adversarial_loss(discriminator(noisy_real, labels), valid)
            
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            gen_imgs = generator(z, labels)
            noisy_fake = add_instance_noise(gen_imgs.detach(), epoch) 
            d_fake_loss = adversarial_loss(discriminator(noisy_fake, labels), fake)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- G ---
            current_g_loss = 0
            for _ in range(N_GENERATOR):
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
                gen_imgs = generator(z, labels)
                
                # Adversarial
                validity = discriminator(gen_imgs, labels)
                g_loss_adv = adversarial_loss(validity, valid)
                
                # ✅ [关键修改] 桥接层 + 强语义引导
                # A. 归一化 [-1, 1] -> [0, 1]
                bridge_imgs = (gen_imgs + 1) * 0.5
                # B. Resize 128x128 -> 39x216
                bridge_resized = torch.nn.functional.interpolate(bridge_imgs, size=(39, 216), mode='bilinear')
                # C. CrossEntropy 计算
                pred_logits = supervisor(bridge_resized)
                g_loss_sem = semantic_criterion(pred_logits, labels)
                
                # ✅ 权重 5.0
                g_loss = g_loss_adv + 5.0 * g_loss_sem
                g_loss.backward()
                optimizer_G.step()
                current_g_loss = g_loss.item()

            g_losses.append(current_g_loss)
            d_losses.append(d_loss.item())

            if i % 20 == 0:
                print(f"[Epoch {epoch}][Batch {i}] D: {d_loss.item():.4f} G: {current_g_loss:.4f}")

        # Epoch End
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.savefig(monitor.paths['overall'] / "total_loss_curve.png")
        plt.close()
        
        if epoch % 5 == 0:
            monitor.micro_eval(epoch)
        
        if epoch > 0 and epoch % MACRO_EVAL_INTERVAL == 0:
            monitor.macro_eval(epoch, real_img_cache)

        torch.save(generator.state_dict(), MODEL_SAVE_DIR / "generator_last.pth")
        print(f"💾 Epoch {epoch} Done.")

if __name__ == "__main__":
    train()