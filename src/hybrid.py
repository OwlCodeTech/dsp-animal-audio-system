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

# ================= 1. 差异化配置中心 =================
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

# ================= 全局超参数 (优化版) =================
BATCH_SIZE = 32
EPOCHS = 100           # ⬆️ 增加轮数，因为正则化会减慢收敛速度
LEARNING_RATE = 0.0001 # ⬇️ 稍微降低学习率，减少曲线毛刺
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径
CSV_PATH = BASE_DIR / "metadata" / "dataset_index.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report_assets/hybrid_emotion_opt"

# ★★★ 预训练 CNN 权重路径 (必须存在) ★★★
CNN_PRETRAINED_PATH = BASE_DIR / "models" / "species_classifier_best.pth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ================= 2. 数据集定义 =================
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
            # 简单的形状校验
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


# ================= 3. 模型组件定义 =================

# --- 组件 A: CNN (老师傅 - 冻结) ---
class SpeciesCNN(nn.Module):
    def __init__(self, num_classes=4):
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
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x) 
        return x

# --- 组件 B: Transformer (新学生 - 优化版) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SpeciesTransformer(nn.Module):
    # ★★★ 修改：默认 dropout 设为 0.4 ★★★
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.4):
        super(SpeciesTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(39, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 在 EncoderLayer 中传入 dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        x = self.input_projection(x)
        x = x * np.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return x

# --- 核心: 混合模型 (Hybrid Fusion - 优化版) ---
class HybridEmotionModel(nn.Module):
    def __init__(self, cnn_pretrained_path, num_classes, d_model=128):
        super(HybridEmotionModel, self).__init__()
        
        # 1. CNN 分支 (Frozen)
        self.cnn = SpeciesCNN(num_classes=4)
        if os.path.exists(cnn_pretrained_path):
            print(f"   🔗 [Hybrid] Loading Pretrained CNN: {Path(cnn_pretrained_path).name}")
            state_dict = torch.load(cnn_pretrained_path, map_location='cpu')
            state_dict = {k:v for k,v in state_dict.items() if 'classifier' not in k}
            self.cnn.load_state_dict(state_dict, strict=False)
        else:
            print("   ⚠️ [Hybrid] Pretrained CNN not found! Random Init.")

        self.cnn.classifier = nn.Flatten()
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # 2. Transformer 分支 (Trainable)
        # ★★★ 启用高 Dropout (0.4) ★★★
        self.transformer = SpeciesTransformer(d_model=d_model, dropout=0.4)
        
        # 3. 融合分类头
        # ★★★ 启用高 Dropout (0.5) ★★★
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # 强力抗过拟合
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            feat_cnn = self.cnn(x) 
        feat_trans = self.transformer(x)
        combined = torch.cat((feat_cnn, feat_trans), dim=1)
        out = self.fusion_head(combined)
        return out


# ================= 4. 辅助绘图函数 =================
def save_plots(history, cm, classes, file_prefix, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{file_prefix} - Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{file_prefix} - Accuracy')
    plt.legend()
    
    plt.savefig(save_dir / f"{file_prefix}_loss_acc_curve.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    label_indices = list(range(len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{file_prefix} - Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    plt.savefig(save_dir / f"{file_prefix}_confusion_matrix.png")
    plt.close()


# ================= 5. 单物种训练流程 =================

def train_one_species(species_name):
    sp_config = SPECIES_CONFIG.get(species_name, DEFAULT_CONFIG)
    noise_level = sp_config.get('noise_level', 0.02)
    manual_weights_dict = sp_config.get('manual_weights', None)

    # 命名处理: 文件夹保持原样，文件名添加前缀
    dir_name = species_name.replace(" ", "_")
    file_prefix = dir_name

    species_model_dir = MODEL_DIR / dir_name
    species_report_dir = REPORT_DIR / dir_name
    
    species_model_dir.mkdir(parents=True, exist_ok=True)
    species_report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 开始优化版混合模型训练: {species_name}")
    print(f"📂 权重目录: {species_model_dir}")
    print(f"⚙️  策略: 噪声={noise_level}, 权重均衡={manual_weights_dict}")
    print(f"{'='*60}")

    train_ds = EmotionDataset(CSV_PATH, species_name, 'Train', augment=True, noise_level=noise_level)
    val_ds = EmotionDataset(CSV_PATH, species_name, 'Val')
    test_ds = EmotionDataset(CSV_PATH, species_name, 'Test')
    
    if len(train_ds.unique_emotions) < 2 or len(train_ds) == 0:
        print(f"⚠️ 跳过 {species_name}: 数据不足。")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    num_emotions = len(train_ds.label_map)
    model = HybridEmotionModel(
        cnn_pretrained_path=CNN_PRETRAINED_PATH, 
        num_classes=num_emotions
    ).to(DEVICE)
    
    # 权重设置
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

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # ★★★ 优化器修改：添加 weight_decay (L2正则化) ★★★
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE,
        weight_decay=1e-4 # 抑制过拟合，平滑曲线
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 训练循环
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
        
        # 文件名保持前缀逻辑
        torch.save(model.state_dict(), species_model_dir / f"{file_prefix}_last.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), species_model_dir / f"{file_prefix}_best.pth")
            
    print(f"\n   ✅ Best Val Acc: {best_acc:.2f}%")

    # 测试与报告
    best_model_path = species_model_dir / f"{file_prefix}_best.pth"
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
            save_plots(history, cm, classes, file_prefix, species_report_dir)
            print(f"   📊 Plots saved to {species_report_dir}")

if __name__ == "__main__":
    TARGET_SPECIES = ['monkey', 'dog', 'deer', 'Gallus gallus']
    
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
    else:
        if CNN_PRETRAINED_PATH.exists():
            print(f"Found Teacher Model (CNN): {CNN_PRETRAINED_PATH}")
        else:
            print(f"Warning: Teacher Model not found. CNN branch will be random.")

        for sp in TARGET_SPECIES:
            train_one_species(sp)