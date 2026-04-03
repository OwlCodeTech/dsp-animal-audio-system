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

# 尝试导入配置
try:
    from src import config
except ImportError:
    import config

# ================= 1. 差异化配置中心 (核心修改) =================
# 针对每种动物的“病症”开出不同的“药方”

SPECIES_CONFIG = {
    # 🐶 狗：之前训练集准确率低 -> 原因是噪声太大 -> 降低噪声
    'dog': {
        'noise_level': 0.005,  # 降低噪声 (原0.02)
        'manual_weights': None # 使用自动权重
    },
    
    # 🐱 猫：之前全猜 Angry -> 原因是样本极度不平衡 -> 使用自动权重惩罚
    'cat': {
        'noise_level': 0.02,
        'manual_weights': None # 自动计算权重足以解决
    },
    
    # 🦌 鹿：Alarm 识别率低 -> 原因是 Alarm 样本太少 -> 使用自动权重
    'deer': {
        'noise_level': 0.02,
        'manual_weights': None
    },
    
    # 🐔 原鸡：Song 被 Alarm 吃掉 -> 原因是特征太像 -> 手动强力纠偏
    # 假设标签顺序是 [alarm, call, song] (字母序)
    # 我们给 Song (index 2) 极为权重，给 Alarm (index 0) 较低权重
    'Gallus gallus': {
        'noise_level': 0.02,
        'manual_weights': {0: 0.5, 1: 1.0, 2: 3.0} # 强行关注 Song
    }
}

# 通用默认配置
DEFAULT_CONFIG = {'noise_level': 0.02, 'manual_weights': None}

# 全局超参数
BATCH_SIZE = 32
EPOCHS = 100           # 增加轮数，因为我们要用小学习率微调
LEARNING_RATE = 0.0001 # 降低学习率 (微调模式)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径
CSV_PATH = BASE_DIR / "metadata" / "dataset_index.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report_assets/transformer_emotion"
PRETRAINED_MODEL_PATH = BASE_DIR / "models" / "species_classifier_best.pth"

# 确保输出目录存在
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ================= 2. 数据集定义 (支持动态噪声) =================

class EmotionDataset(Dataset):
    def __init__(self, csv_path, species_name, split_type='Train', augment=False, noise_level=0.02):
        self.augment = augment
        self.noise_level = noise_level # 接收动态噪声参数
        
        df = pd.read_csv(csv_path)
        
        # 大小写不敏感匹配
        df['Species_Lower'] = df['Species'].str.lower().str.strip()
        target_species_lower = species_name.lower().strip()
        
        self.df = df[
            (df['Species_Lower'] == target_species_lower) & 
            (df['Split'] == split_type)
        ].reset_index(drop=True)
        
        self.unique_emotions = sorted(self.df['Emotion'].unique())
        self.label_map = {name: idx for idx, name in enumerate(self.unique_emotions)}
        
        if split_type == 'Train' and len(self.df) > 0:
            print(f"   > 标签映射: {self.label_map}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]['FeaturePath']
        rel_path = rel_path.replace('./', '').replace('/', os.sep).replace('\\', os.sep)
        npy_path = BASE_DIR / rel_path
        
        try:
            features = np.load(npy_path)
        except Exception:
            features = np.zeros((39, 216), dtype=np.float32)

        features_tensor = torch.from_numpy(features).float()

        # 数据增强：使用传入的 noise_level
        if self.augment:
            noise = torch.randn_like(features_tensor) * self.noise_level
            features_tensor += noise

        features_tensor = features_tensor.unsqueeze(0)
        
        emotion_name = self.df.iloc[idx]['Emotion']
        label = self.label_map[emotion_name]
        
        return features_tensor, label

# ================= 3. 模型定义 =================

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
    def __init__(self, num_classes, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(SpeciesTransformer, self).__init__()
        self.d_model = d_model
        
        # 输入投影层：将39维MFCC特征映射到d_model维
        self.input_projection = nn.Linear(39, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局平均池化和分类器
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 1, 39, 216)
        batch_size = x.size(0)
        
        # 移除通道维度并转置：(batch_size, 216, 39)
        x = x.squeeze(1).transpose(1, 2)
        
        # 输入投影：(batch_size, 216, d_model)
        x = self.input_projection(x)
        
        # 缩放输入
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, 216, d_model)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # (batch_size, d_model, 216)
        x = self.global_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        # 分类
        x = self.classifier(x)
        return x

# 保留原CNN类作为备选
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

# ================= 4. 辅助绘图函数 =================

def save_plots(history, cm, classes, species, save_dir):
    # 1. 曲线图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{species} - Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{species} - Accuracy')
    plt.legend()
    plt.savefig(save_dir / "loss_acc_curve.png")
    plt.close()

    # 2. 混淆矩阵
    plt.figure(figsize=(8, 6))
    # 确保 labels 完整，防止缺类报错
    label_indices = list(range(len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{species} - Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()

# ================= 5. 单物种训练流程 (集成解决方案) =================

def train_one_species(species_name):
    # 0. 获取该物种的专属配置
    sp_config = SPECIES_CONFIG.get(species_name, DEFAULT_CONFIG)
    noise_level = sp_config.get('noise_level', 0.02)
    manual_weights_dict = sp_config.get('manual_weights', None)

    # 路径处理
    safe_species_name = species_name.replace(" ", "_")
    species_model_dir = MODEL_DIR / safe_species_name
    species_report_dir = REPORT_DIR / safe_species_name
    species_model_dir.mkdir(parents=True, exist_ok=True)
    species_report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*40}")
    print(f"🚀 开始训练: {species_name}")
    print(f"⚙️  策略配置: 噪声强度={noise_level}, 手动权重={manual_weights_dict is not None}")
    print(f"📊 模型架构: CNN (2D卷积神经网络)")
    print(f"{'='*40}")

    # 1. 准备数据集 (传入定制 noise_level)
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
    
    # 2. 模型初始化 - 使用CNN架构
    num_emotions = len(train_ds.label_map)
    model = SpeciesCNN(num_classes=num_emotions).to(DEVICE)
    
    # 3. 损失函数权重计算 (解决类别不平衡)
    if manual_weights_dict:
        # 方案 A: 使用手动指定的权重 (针对 Gallus gallus)
        # 确保按 label_map 的顺序排列权重
        weights_list = []
        for name, idx in train_ds.label_map.items():
            w = manual_weights_dict.get(idx, 1.0) # 默认 1.0
            weights_list.append(w)
        class_weights = torch.FloatTensor(weights_list).to(DEVICE)
        print(f"   🔧 使用手动权重: {weights_list}")
    else:
        # 方案 B: 自动计算平衡权重 (针对 Cat, Deer)
        y_train = train_ds.df['Emotion'].map(train_ds.label_map).values
        # compute_class_weight 可能会返回 numpy float64
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
        print(f"   ⚖️ 自动计算权重: {class_weights.cpu().numpy().round(2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器 (使用小学习率微调)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 5. 训练循环
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    epoch_pbar = tqdm(range(EPOCHS), desc=f"Training {species_name}", unit="epoch")
    
    for epoch in epoch_pbar:
        # Train
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
        
        # Val
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
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 监控学习率变化
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0:
            print(f"   当前学习率: {current_lr:.6f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_pbar.set_postfix({"Tr_Acc": f"{train_acc:.1f}%", "Val_Acc": f"{val_acc:.1f}%", "LR": f"{current_lr:.6f}"})
        
        # 保存权重
        torch.save(model.state_dict(), species_model_dir / "last.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), species_model_dir / "best.pth")
            
    print(f"\n   ✅ Best Val Acc: {best_acc:.2f}%")

    # 6. 测试与报告
    if (species_model_dir / "best.pth").exists():
        model.load_state_dict(torch.load(species_model_dir / "best.pth"))
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                out = model(x)
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.numpy())
        
        # 修正：classification_report 必须包含 labels 参数
        if len(all_labels) > 0:
            classes = list(train_ds.label_map.keys())
            label_indices = list(range(len(classes)))
            
            print(f"\n--- {species_name} Report ---")
            print(classification_report(all_labels, all_preds, labels=label_indices, target_names=classes, zero_division=0))
            
            cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
            save_plots(history, cm, classes, safe_species_name, species_report_dir)
            print(f"   📊 Plots saved to {species_report_dir}")
        else:
            print("   ⚠️ 测试集为空，跳过评估。")

# ================= 主程序 =================

if __name__ == "__main__":
    # 需要训练的物种列表 (大小写不敏感，但建议写对)
    TARGET_SPECIES = ['monkey', 'dog', 'deer', 'Gallus gallus']
    
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
    else:
        for sp in TARGET_SPECIES:
            train_one_species(sp)