# src/gui_shared.py
import sys
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QSlider, QStyle
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# --- 1. 路径定位 ---
BASE_DIR = Path(__file__).resolve().parent.parent

STATS_CSV_PATH = BASE_DIR / "metadata" / "species_statistics.csv"
INDEX_CSV_PATH = BASE_DIR / "metadata" / "dataset_index.csv"
MODEL_PATH = BASE_DIR / "models" / "species_classifier_best.pth"

# [关键] 报告/图片素材目录
REPORT_DIR = BASE_DIR / "report_assets"

# --- 2. 动态加载标签 ---
try:
    if INDEX_CSV_PATH.exists():
        import pandas as pd
        df_index = pd.read_csv(INDEX_CSV_PATH)
        df_train = df_index[df_index['Split'] == 'Train']
        SPECIES_LABELS = sorted(df_train['Species'].astype(str).unique().tolist())
    else:
        SPECIES_LABELS = ['cat', 'deer', 'dog', 'dolphin', 'elephant', 'Gallus gallus', 'sperm whale']
except Exception as e:
    print(f"❌ 标签加载失败: {e}")
    SPECIES_LABELS = ['cat', 'deer', 'dog', 'dolphin', 'elephant', 'Gallus gallus', 'sperm whale']

# --- 3. 全局样式表 ---
GLOBAL_STYLES = """
    QMainWindow { background-color: #F5F7FA; }
    QTabWidget::pane { border: 1px solid #E0E0E0; background: #FFFFFF; border-radius: 8px; margin: 10px;}
    QTabBar::tab { 
        background: #EAEAEA; padding: 10px 30px; border-top-left-radius: 6px; border-top-right-radius: 6px;
        font-family: "Microsoft YaHei"; font-weight: bold; color: #666; margin-right: 2px;
    }
    QTabBar::tab:selected { background: #FFFFFF; color: #007AFF; border-top: 3px solid #007AFF; }
    
    QGroupBox { 
        font-weight: bold; border: 1px solid #DDD; border-radius: 8px; 
        margin-top: 24px; 
        background: #FFFFFF; 
    }
    QGroupBox::title { 
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 15px; top: 0px; padding: 0 5px; color: #007AFF; background: transparent; 
    }
    
    QPushButton {
        background-color: #FFFFFF; border: 1px solid #D0D0D0; border-radius: 6px;
        padding: 8px; font-family: "Microsoft YaHei"; color: #333;
    }
    QPushButton:hover { background-color: #F0F8FF; border-color: #007AFF; color: #007AFF; }
    QPushButton:pressed { background-color: #E1F0FF; }
    QPushButton:disabled { background-color: #F5F5F5; color: #BBB; border: 1px solid #EEE; }
    
    QPushButton.primary { background-color: #007AFF; color: white; border: none; }
    QPushButton.primary:hover { background-color: #0056b3; }
"""

# --- 4. 自定义播放器组件 (保持不变) ---
class AudioPlayerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.player = QMediaPlayer()
        self.init_ui()
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)
        self.player.mediaStatusChanged.connect(self.handle_media_status)
        
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8) 
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setFixedSize(32, 32) 
        self.btn_play.setStyleSheet("QPushButton { background-color: #007AFF; border-radius: 16px; border: none; } QPushButton:hover { background-color: #0056b3; } QPushButton:disabled { background-color: #dcdcdc; }")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        time_style = "color: #666; font-family: Consolas, monospace; font-size: 11px;"
        self.lbl_current = QLabel("00:00")
        self.lbl_current.setStyleSheet(time_style)
        self.lbl_total = QLabel("00:00")
        self.lbl_total.setStyleSheet(time_style)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet("QSlider::groove:horizontal { height: 4px; background: #e0e0e0; border-radius: 2px; } QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0; background: #007AFF; border-radius: 6px; }")
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        self.progress_container = QWidget()
        pc_layout = QHBoxLayout(self.progress_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.addWidget(self.lbl_current)
        pc_layout.addWidget(self.slider)
        pc_layout.addWidget(self.lbl_total)
        self.progress_container.setVisible(False)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.progress_container)
        self.setLayout(layout)
        
    def set_media(self, file_path):
        self.player.stop()
        url = QUrl.fromLocalFile(str(file_path))
        self.player.setMedia(QMediaContent(url))
        self.btn_play.setEnabled(True)
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.progress_container.setVisible(True)
        
    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.player.play()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            
    def handle_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.player.stop()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.slider.setValue(0)
            self.lbl_current.setText("00:00")

    def update_position(self, position):
        self.slider.setValue(position)
        self.lbl_current.setText(self.format_time(position))
        
    def update_duration(self, duration):
        self.slider.setRange(0, duration)
        self.lbl_total.setText(self.format_time(duration))
        
    def set_position(self, position):
        self.player.setPosition(position)
        
    def format_time(self, ms):
        seconds = (ms // 1000) % 60
        minutes = (ms // 60000)
        return f"{minutes:02}:{seconds:02}"