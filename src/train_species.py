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

# ================= 配置参数 =================
BATCH_SIZE = 32         
EPOCHS = 80             
LEARNING_RATE = 0.001   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径
CSV_PATH = BASE_DIR / "metadata" / "dataset_index.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report_assets"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 1. 数据集定义 =================

class SpeciesDataset(Dataset):
    def __init__(self, csv_path, split_type='Train', label_map=None, augment=False):
        self.augment = augment
        self.noise_level = 0.015

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['Split'] == split_type].reset_index(drop=True)
        
        if label_map is None:
            unique_species = sorted(self.df['Species'].unique())
            self.label_map = {name: idx for idx, name in enumerate(unique_species)}
        else:
            self.label_map = label_map
            
        print(f"[{split_type}] Set Loaded: {len(self.df)} samples.")
        if split_type == 'Train':
            print(f"Classes Map: {self.label_map}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]['FeaturePath']
        rel_path = rel_path.replace('./', '').replace('/', os.sep)
        npy_path = BASE_DIR / rel_path
        
        # --- [新增] 稳健加载逻辑 ---
        try:
            features = np.load(npy_path)
            if features.shape != (39, 216):
                 raise ValueError("Shape mismatch")
        except Exception as e:
            # 遇到坏数据，返回零噪声
            features = np.random.randn(39, 216).astype(np.float32) * 0.001

        features_tensor = torch.from_numpy(features).float()

        if self.augment:
            noise = torch.randn_like(features_tensor) * self.noise_level
            features_tensor += noise

        features_tensor = features_tensor.unsqueeze(0)
        
        species_name = self.df.iloc[idx]['Species']
        label = self.label_map[species_name]
        
        return features_tensor, label

# ================= 2. 模型定义 (基座) =================

class SpeciesCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpeciesCNN, self).__init__()
        
        # 3层 CNN，足以提取 MFCC 特征
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

# ================= 3. 辅助绘图 =================

def plot_history(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(REPORT_DIR / "training_history_species.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    label_indices = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig(REPORT_DIR / "confusion_matrix_species.png")
    plt.close()

# ================= 4. 主训练流程 =================

def train_pipeline():
    print(f"Using Device: {DEVICE}")
    
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    train_dataset = SpeciesDataset(CSV_PATH, 'Train', augment=True)
    val_dataset = SpeciesDataset(CSV_PATH, 'Val', label_map=train_dataset.label_map)
    test_dataset = SpeciesDataset(CSV_PATH, 'Test', label_map=train_dataset.label_map)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(train_dataset.label_map)
    classes = list(train_dataset.label_map.keys())
    
    # 类别平衡权重
    y_train = train_dataset.df['Species'].map(train_dataset.label_map).values
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Class Weights: {class_weights.cpu().numpy().round(2)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = SpeciesCNN(num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            r_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = r_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = v_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_epoch_acc = 100 * v_correct / v_total if v_total > 0 else 0
        
        scheduler.step(val_epoch_loss)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}: Loss={val_epoch_loss:.4f}, Acc={val_epoch_acc:.2f}%")
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), MODEL_DIR / "species_classifier_best.pth")

    plot_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
    
    # 测试
    if (MODEL_DIR / "species_classifier_best.pth").exists():
        model.load_state_dict(torch.load(MODEL_DIR / "species_classifier_best.pth"))
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        label_indices = list(range(len(classes)))
        print(classification_report(all_labels, all_preds, labels=label_indices, target_names=classes, zero_division=0))
        plot_confusion_matrix(all_labels, all_preds, classes)

if __name__ == "__main__":
    train_pipeline()