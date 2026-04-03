import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

# --- 路径定位 ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# ================= 1. 差异化配置中心 (保留当前代码的配置) =================
SPECIES_CONFIG = {
    'dog': {
        'noise_level': 0.005,  
        'manual_weights': {0: 2.5, 1: 1.2, 2: 1.0} 
    },
    'monkey': {
        'noise_level': 0.015,
        'manual_weights': {0: 2.0, 1: 2.0, 2: 0.8}
    },
    'deer': {
        'noise_level': 0.01,
        'manual_weights': {0: 3.5, 1: 1.0, 2: 1.2}
    },
    'Gallus gallus': {
        'noise_level': 0.015,
        'manual_weights': {0: 2.5, 1: 2.0, 2: 0.8} 
    }
}

DEFAULT_CONFIG = {'noise_level': 0.02, 'manual_weights': None}

# 全局超参数 (保留当前代码的设置)
BATCH_SIZE = 32
EPOCHS = 80            
LEARNING_RATE = 0.00015 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径
CSV_PATH = BASE_DIR / "metadata" / "dataset_index.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report_assets"

# 预训练权重路径
PRETRAINED_MODEL_PATH = BASE_DIR / "models" / "species_classifier_best.pth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ================= 2. 数据集定义 (保留当前代码的逻辑) =================
class EmotionDataset(Dataset):
    def __init__(self, csv_path, species_name, split_type='Train', augment=False, noise_level=0.02):
        self.augment = augment
        self.noise_level = noise_level 
        
        df = pd.read_csv(csv_path)
        df['Species_Lower'] = df['Species'].str.lower().str.strip()
        target_species_lower = species_name.lower().strip()
        
        self.df = df[
            (df['Species_Lower'] == target_species_lower) & 
            (df['Split'] == split_type)
        ].reset_index(drop=True)
        
        self.unique_emotions = sorted(self.df['Emotion'].unique())
        self.label_map = {name: idx for idx, name in enumerate(self.unique_emotions)}
        
        if split_type == 'Train' and len(self.df) > 0:
            print(f"   > [{species_name}] 标签映射: {self.label_map}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]['FeaturePath']
        rel_path = rel_path.replace('./', '').replace('/', os.sep).replace('\\', os.sep)
        npy_path = BASE_DIR / rel_path
        
        try:
            features = np.load(npy_path)
            if features.shape != (39, 216):
                 pass 
        except Exception:
            features = np.zeros((39, 216), dtype=np.float32)

        features_tensor = torch.from_numpy(features).float()
        if self.augment:
            noise = torch.randn_like(features_tensor) * self.noise_level
            features_tensor += noise
        features_tensor = features_tensor.unsqueeze(0)
        
        emotion_name = self.df.iloc[idx]['Emotion']
        label = self.label_map[emotion_name]
        
        return features_tensor, label


# ================= 3. 模型定义 (保留当前代码的逻辑) =================
class SpeciesCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpeciesCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


# ================= 4. 辅助绘图函数 (逻辑修改为 Code A 风格) =================
def save_plots(history, cm, classes, species_name, save_dir):
    """
    修改为 Code A 风格：
    1. 移除文件名前缀参数
    2. 只有 Title 使用 species_name
    3. 文件名固定为 loss_acc_curve.png 和 confusion_matrix.png
    """
    # 1. 曲线图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{species_name} - Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{species_name} - Accuracy')
    plt.legend()
    
    # [修改] 移除前缀，直接保存
    plt.savefig(save_dir / "loss_acc_curve.png")
    plt.close()

    # 2. 混淆矩阵
    plt.figure(figsize=(8, 6))
    label_indices = list(range(len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{species_name} - Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    # [修改] 移除前缀，直接保存
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()


# ================= 5. 单物种训练流程 =================

def train_one_species(species_name):
    sp_config = SPECIES_CONFIG.get(species_name, DEFAULT_CONFIG)
    noise_level = sp_config.get('noise_level', 0.02)
    manual_weights_dict = sp_config.get('manual_weights', None)

    # 1. 文件夹名称
    safe_species_name = species_name.replace(" ", "_")
    
    species_model_dir = MODEL_DIR / safe_species_name
    species_report_dir = REPORT_DIR / safe_species_name
    
    species_model_dir.mkdir(parents=True, exist_ok=True)
    species_report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 开始训练: {species_name}")
    print(f"📂 权重目录: {species_model_dir}")
    print(f"🖼️  图片目录: {species_report_dir}")
    print(f"⚙️  策略: 噪声={noise_level}, 权重均衡={manual_weights_dict}")
    print(f"{'='*60}")

    # 1. 准备数据集
    train_ds = EmotionDataset(CSV_PATH, species_name, 'Train', augment=True, noise_level=noise_level)
    val_ds = EmotionDataset(CSV_PATH, species_name, 'Val')
    test_ds = EmotionDataset(CSV_PATH, species_name, 'Test')
    
    if len(train_ds.unique_emotions) < 2:
        print(f"⚠️ 跳过 {species_name}: 只有 {len(train_ds.unique_emotions)} 种情绪。")
        return
    if len(train_ds) == 0:
        print(f"⚠️ 跳过 {species_name}: 训练集为空。")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 2. 模型初始化
    num_emotions = len(train_ds.label_map)
    model = SpeciesCNN(num_classes=num_emotions).to(DEVICE)
    
    # 迁移学习加载 (保留当前代码的逻辑)
    if PRETRAINED_MODEL_PATH.exists():
        print(f"   📥 加载预训练权重: {PRETRAINED_MODEL_PATH.name}")
        pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        # 解冻策略 (保留当前代码的逻辑)
        if species_name in ['dog', 'monkey', 'Gallus gallus']:
            print(f"   🔓 难例微调: 冻结 Conv1 -> 解冻 Conv2, Conv3, Classifier")
            for name, param in model.named_parameters():
                if "conv1" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            print(f"   🔓 标准微调: 冻结 Conv1 -> 解冻 Conv2, Conv3, Classifier")
            for name, param in model.named_parameters():
                if "conv1" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    else:
        print("   ⚠️ 未找到预训练模型，使用随机初始化。")

    # 3. 权重设置
    if manual_weights_dict:
        weights_list = []
        for name, idx in train_ds.label_map.items():
            w = manual_weights_dict.get(idx, 1.0)
            weights_list.append(w)
        class_weights = torch.FloatTensor(weights_list).to(DEVICE)
    else:
        y_train = train_ds.df['Emotion'].map(train_ds.label_map).values
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)

    # 损失函数 (保留当前代码逻辑：普通的 CrossEntropy，无 Label Smoothing)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 5. 训练循环
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    epoch_pbar = tqdm(range(EPOCHS), desc=f"Training {species_name}", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            r_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        train_loss = r_loss / len(train_loader)
        train_acc = 100 * correct / total if total > 0 else 0
        
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item()
                _, pred = torch.max(out, 1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
        
        val_loss = v_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * v_correct / v_total if v_total > 0 else 0
        
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_pbar.set_postfix({"Tr_Acc": f"{train_acc:.1f}%", "Val_Acc": f"{val_acc:.1f}%", "Best": f"{best_acc:.1f}%"})
        
        # [修改] 文件命名逻辑改为 Code A 样式 (last.pth)
        torch.save(model.state_dict(), species_model_dir / "last.pth")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # [修改] 文件命名逻辑改为 Code A 样式 (best.pth)
            torch.save(model.state_dict(), species_model_dir / "best.pth")
            
    print(f"\n   ✅ Best Val Acc: {best_acc:.2f}%")

    # 6. 测试与报告
    # [修改] 加载 best.pth (无前缀)
    best_model_path = species_model_dir / "best.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                out = model(x)
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.numpy())
        
        if len(all_labels) > 0:
            classes = list(train_ds.label_map.keys())
            label_indices = list(range(len(classes)))
            print(classification_report(all_labels, all_preds, labels=label_indices, target_names=classes, zero_division=0))
            
            cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
            
            # [修改] 调用 save_plots 时传入 safe_species_name (仅用于标题), 内部使用固定文件名
            save_plots(history, cm, classes, safe_species_name, species_report_dir)
            print(f"   📊 Plots saved to {species_report_dir}")

if __name__ == "__main__":
    TARGET_SPECIES = ['monkey', 'dog', 'deer', 'Gallus gallus']
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
    else:
        if PRETRAINED_MODEL_PATH.exists():
            print(f"Found pretrained species model at: {PRETRAINED_MODEL_PATH}")
        else:
            print(f"Warning: Pretrained model not found. Training from scratch.")

        for sp in TARGET_SPECIES:
            train_one_species(sp)