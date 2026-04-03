# src/tab_training.py
import os
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QListWidget, 
                             QLabel, QFrame, QSizePolicy, QSplitter, QToolButton)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt, QSize, QEvent

# 导入公共配置
from gui_shared import BASE_DIR, REPORT_DIR

# --- 自定义组件：自动等比缩放的图片标签 ---
class AutoResizingLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        # 忽略尺寸策略，完全由布局控制大小
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(50, 50)
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_image()

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

    def update_image(self):
        if self._pixmap and not self._pixmap.isNull():
            size = self.size()
            # 保持比例缩放，确保图片不变形
            scaled_pixmap = self._pixmap.scaled(
                size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)
        else:
            super().setText("暂无图片")

# --- 主界面类 ---
class TrainingVizTab(QWidget):
    def __init__(self):
        super().__init__()
        self.is_drawer_open = True
        self.init_ui()
        self.init_data()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ================= 1. 左侧抽屉 =================
        self.drawer_container = QFrame()
        self.drawer_container.setFixedWidth(240)
        self.drawer_container.setStyleSheet("background: #f8f9fa; border-right: 1px solid #e0e0e0;")
        drawer_layout = QVBoxLayout(self.drawer_container)
        drawer_layout.setContentsMargins(0, 0, 0, 0)
        drawer_layout.setSpacing(0)

        # 1.1 顶部栏
        header_bar = QFrame()
        header_bar.setFixedHeight(50)
        header_bar.setStyleSheet("background: #fff; border-bottom: 1px solid #eee;")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(10, 0, 10, 0)
        
        self.toggle_btn = QToolButton()
        self.toggle_btn.setArrowType(Qt.DownArrow)
        self.toggle_btn.setFixedSize(30, 30)
        self.toggle_btn.setStyleSheet("border: none; border-radius: 4px;")
        self.toggle_btn.clicked.connect(self.toggle_drawer)
        
        self.drawer_title = QLabel("模型库")
        self.drawer_title.setStyleSheet("font-weight: bold; color: #555;")
        
        header_layout.addWidget(self.toggle_btn)
        header_layout.addWidget(self.drawer_title)
        header_layout.addStretch()

        # 1.2 列表
        self.model_list = QListWidget()
        self.model_list.setFrameShape(QFrame.NoFrame)
        self.model_list.setStyleSheet("""
            QListWidget { background: transparent; outline: none; font-size: 13px; }
            QListWidget::item { padding: 12px 15px; color: #333; border-bottom: 1px solid #f0f0f0; }
            QListWidget::item:selected { background: #e3f2fd; color: #007AFF; border-left: 3px solid #007AFF; }
            QListWidget::item:hover:!selected { background: #f5f5f5; }
        """)
        self.model_list.currentRowChanged.connect(self.on_model_selected)

        drawer_layout.addWidget(header_bar)
        drawer_layout.addWidget(self.model_list)

        # ================= 2. 右侧内容区 =================
        content_container = QWidget()
        content_container.setStyleSheet("background: #ffffff;")
        content_layout = QVBoxLayout(content_container)
        # [优化] 减小边距，给图片更多空间
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        # 2.1 标题
        self.lbl_model_name = QLabel("请选择模型查看分析结果")
        self.lbl_model_name.setAlignment(Qt.AlignCenter)
        self.lbl_model_name.setFixedHeight(30)
        self.lbl_model_name.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #333;
            background: transparent; margin-bottom: 5px;
        """)
        
        # 2.2 图表容器 (水平布局)
        charts_container = QWidget()
        charts_layout = QHBoxLayout(charts_container)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(10) # 减小图片间距

        self.box_cm, self.img_cm = self.create_chart_card("🔥 混淆矩阵")
        self.box_loss, self.img_loss = self.create_chart_card("📉 训练曲线")

        charts_layout.addWidget(self.box_cm, 1)
        charts_layout.addWidget(self.box_loss, 1)

        content_layout.addWidget(self.lbl_model_name, 0)
        content_layout.addWidget(charts_container, 1) 

        # ================= 3. 组合 (Splitter) =================
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.drawer_container)
        self.splitter.addWidget(content_container)
        self.splitter.setCollapsible(0, False) 
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("QSplitter::handle { background: #e0e0e0; }")

        main_layout.addWidget(self.splitter)

    def create_chart_card(self, title_text):
        """创建无边框、仅阴影的极简卡片"""
        card = QFrame()
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # [优化] 去掉边框，只保留白色背景，视觉上消除“空缺感”
        card.setStyleSheet("""
            QFrame {
                background: white; 
                border-radius: 8px;
                border: 1px solid #f0f0f0; 
            }
        """)
        
        # 淡淡的阴影
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        from PyQt5.QtGui import QColor
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0,0,0,10)) # 非常淡的阴影
        shadow.setOffset(0, 2)
        card.setGraphicsEffect(shadow)

        layout = QVBoxLayout(card)
        # [优化] 极小内边距，让图片尽可能撑满
        layout.setContentsMargins(5, 10, 5, 5)
        layout.setSpacing(5)
        
        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setFixedHeight(20)
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #555; border: none;")
        
        img_label = AutoResizingLabel()
        img_label.setStyleSheet("border: none;") # 图片本身无边框
        
        layout.addWidget(title)
        layout.addWidget(img_label, 1) 
        
        return card, img_label

    def toggle_drawer(self):
        self.is_drawer_open = not self.is_drawer_open
        if self.is_drawer_open:
            self.drawer_container.setFixedWidth(240)
            self.model_list.show()
            self.drawer_title.show()
            self.toggle_btn.setArrowType(Qt.DownArrow)
        else:
            self.drawer_container.setFixedWidth(48)
            self.model_list.hide()
            self.drawer_title.hide()
            self.toggle_btn.setArrowType(Qt.RightArrow)

    def init_data(self):
        self.models_map = {
            "🦁 物种分类器 (Species)": "species",
            "🐱 猫情绪模型 (Cat)": "cat",
            "🐶 狗情绪模型 (Dog)": "dog",
            "🦌 鹿情绪模型 (Deer)": "deer",
            "🐓 原鸡情绪模型 (Gallus)": "Gallus_gallus"
        }
        for name in self.models_map.keys():
            self.model_list.addItem(name)

    def on_model_selected(self, row):
        if row < 0: return
        name = self.model_list.item(row).text()
        key = self.models_map[name]
        self.lbl_model_name.setText(f"{name} 训练报告")
        
        if key == "species":
            cm_path = REPORT_DIR / "confusion_matrix_species.png"
            loss_path = REPORT_DIR / "training_history_species.png"
        else:
            subdir = REPORT_DIR / key
            cm_path = subdir / "confusion_matrix.png"
            loss_path = subdir / "loss_acc_curve.png"
            
        self.update_image(cm_path, self.img_cm)
        self.update_image(loss_path, self.img_loss)

    def update_image(self, path, label):
        if path.exists():
            pixmap = QPixmap(str(path))
            label.setPixmap(pixmap)
            label.setStyleSheet("border: none;")
        else:
            label.setPixmap(None)
            label.setText("未找到图片\n请确认训练已完成")
            label.setStyleSheet("border: 2px dashed #ffcccc; color: red;")