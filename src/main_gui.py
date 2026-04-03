# src/main_gui.py
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QFrame
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# [修改] 导入所有模块 (去掉 src.)
from gui_shared import GLOBAL_STYLES, BASE_DIR
from tab_analysis import AnalysisTab
from tab_interaction import InteractionTab
from tab_training import TrainingVizTab  # [新增]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动物声学翻译与交互系统 (DSP + AI)")
        
        screen = QApplication.primaryScreen().geometry()
        w = int(screen.width() * 0.85)  
        h = int(screen.height() * 0.75)
        self.resize(w, h)
        self.move((screen.width() - w) // 2, (screen.height() - h) // 2)

        self.setStyleSheet(GLOBAL_STYLES)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 10, 20, 20)
        
        # Header
        header_box = QFrame()
        header_box.setStyleSheet("background: white; border-radius: 10px; border-bottom: 4px solid #007AFF;")
        hb_layout = QHBoxLayout(header_box)
        title = QLabel("🦁 动物声学行为分析与交互系统")
        title.setStyleSheet("font-size: 26px; font-weight: 900; color: #333; font-family: 'Microsoft YaHei';")
        subtitle = QLabel("DSP 课程大作业 | 深度学习 | 大模型交互")
        subtitle.setStyleSheet("font-size: 14px; color: #888; font-style: italic;")
        hb_layout.addWidget(title)
        hb_layout.addStretch()
        hb_layout.addWidget(subtitle)
        
        layout.addWidget(header_box)
        
        # Tab Widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 初始化子页面
        self.tab1 = AnalysisTab()
        self.tab2 = InteractionTab()
        self.tab3 = TrainingVizTab() # [新增] 初始化训练展示页
        
        # 添加页面
        self.tabs.addTab(self.tab1, "📊 信号分析实验室")
        self.tabs.addTab(self.tab2, "💬 跨物种交互终端")
        self.tabs.addTab(self.tab3, "📈 模型训练表现") # [新增]
        
        # 连接信号
        self.tab1.analysis_completed.connect(self.tab2.update_context)

if __name__ == "__main__":
    # 确保主程序能找到 src 模块
    sys.path.append(str(BASE_DIR))
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())