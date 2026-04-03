# src/tab_interaction.py
import re
import time
import random
import pandas as pd
import torch
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QTextBrowser, QMessageBox, QFrame, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal

# 导入公共组件
from gui_shared import BASE_DIR, INDEX_CSV_PATH, AudioPlayerWidget
from preprocess import AudioCleaner
from feature_extract import FeatureAnalyzer
from train_species import SpeciesCNN
from llm_agent import AnimalLLM
from audio_generator import AudioGenerator
# --- 线程类 (仅属于 Tab 2) ---
class EmotionInferenceThread(QThread):
    finished = pyqtSignal(str)
    def __init__(self, audio_path, species_name):
        super().__init__()
        self.audio_path = audio_path
        self.species_name = species_name
    def run(self):
        detected_emotion = "Unknown"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        safe_name = self.species_name.replace(" ", "_")
        emo_model_path = BASE_DIR / "models" / safe_name / "best.pth"
        
        if not emo_model_path.exists():
            self.finished.emit("Unknown")
            return
        try:
            cleaner = AudioCleaner()
            analyzer = FeatureAnalyzer()
            segs, sr, _ = cleaner.process_single_file_memory(self.audio_path, self.species_name)
            if segs:
                seg = segs[len(segs)//2] if len(segs) > 1 else segs[0]
                feat = analyzer.extract_39d_mfcc(seg, sr)
                inp = torch.from_numpy(feat).float().unsqueeze(0).unsqueeze(0).to(device)
                state = torch.load(emo_model_path, map_location=device)
                num_classes = state['classifier.2.weight'].shape[0]
                model = SpeciesCNN(num_classes=num_classes).to(device)
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    out = model(inp)
                    pred_idx = torch.argmax(out, dim=1).item()
                df = pd.read_csv(INDEX_CSV_PATH)
                df_sp = df[(df['Species'].astype(str).str.lower() == self.species_name.lower()) & (df['Split'] == 'Train')]
                if len(df_sp) > 0:
                    emotions = sorted(df_sp['Emotion'].unique().tolist())
                    if 0 <= pred_idx < len(emotions): detected_emotion = emotions[pred_idx]
        except Exception as e:
            print(f"Emotion predict error: {e}")
        self.finished.emit(detected_emotion)

class LLMWorker(QThread):
    finished = pyqtSignal(str)
    def __init__(self, species, emotion):
        super().__init__()
        self.species = species
        self.emotion = emotion
        self.agent = AnimalLLM()
    def run(self):
        result = self.agent.analyze(self.species, self.emotion)
        self.finished.emit(result)

class AudioGenWorker(QThread):
    finished = pyqtSignal(str)
    def __init__(self, prompt, save_path):
        super().__init__()
        self.prompt = prompt
        self.save_path = save_path
        self.generator = AudioGenerator()
    def run(self):
        success = self.generator.generate(self.prompt, self.save_path)
        if success: self.finished.emit(str(self.save_path))
        else: self.finished.emit("error")

# --- Tab 2 主界面类 ---
class InteractionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_species = None
        self.current_audio_path = None
        self.generated_prompt = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 左侧聊天窗口
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_browser = QTextBrowser()
        self.chat_browser.setStyleSheet("""
            QTextBrowser { 
                background: white; border: 1px solid #ddd; border-radius: 8px; 
                padding: 20px; font-size: 15px; line-height: 1.6;
            }
        """)
        self.chat_browser.setHtml("""
            <div style='text-align:center; color:#888; margin-top:50px;'>
            <h2>🤖 跨物种智能交互终端</h2>
            <p>请先在 <b>[信号分析实验室]</b> 识别出动物种类，<br>然后在此处进行情感推理与双向沟通。</p>
            </div>
        """)
        chat_layout.addWidget(self.chat_browser)
        
        # 右侧控制台
        ctrl_panel = QFrame()
        ctrl_panel.setFixedWidth(180) 
        ctrl_panel.setStyleSheet("background: #FFFFFF; border-left: 1px solid #eee;")
        ctrl_layout = QVBoxLayout(ctrl_panel)
        # 保持修复后的边距
        ctrl_layout.setContentsMargins(10, 10, 10, 10)
        
        lbl_ctrl = QLabel("控制台")
        lbl_ctrl.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 15px; color: #333;")
        
        grp_step1 = QGroupBox("Step 1: 认知")
        l1 = QVBoxLayout()
        self.btn_llm = QPushButton("🧠 启动分析")
        self.btn_llm.setMinimumHeight(40)
        self.btn_llm.setProperty("class", "primary")
        self.btn_llm.clicked.connect(self.run_llm_analysis)
        l1.addWidget(self.btn_llm)
        grp_step1.setLayout(l1)
        
        grp_step2 = QGroupBox("Step 2: 反馈")
        l2 = QVBoxLayout()
        self.btn_gen = QPushButton("🔊 生成音频")
        self.btn_gen.setMinimumHeight(40)
        self.btn_gen.clicked.connect(self.run_audio_generation)
        self.btn_gen.setEnabled(False)
        lbl_player = QLabel("试听:")
        lbl_player.setStyleSheet("color: #666; font-size: 12px; margin-top: 10px;")
        self.player_widget = AudioPlayerWidget()
        l2.addWidget(self.btn_gen)
        l2.addWidget(lbl_player)
        l2.addWidget(self.player_widget)
        grp_step2.setLayout(l2)
        
        ctrl_layout.addWidget(lbl_ctrl)
        ctrl_layout.addWidget(grp_step1)
        ctrl_layout.addSpacing(15)
        ctrl_layout.addWidget(grp_step2)
        ctrl_layout.addStretch()
        
        layout.addWidget(chat_panel, 1)
        layout.addWidget(ctrl_panel)

    # --- 槽函数：接收来自 Tab 1 的数据 ---
    def update_context(self, species, audio_path):
        self.current_species = species
        self.current_audio_path = audio_path
        
        self.chat_browser.setHtml(f"""
            <div style='text-align:center; margin-top:20px;'>
            <h3>✅ 目标已锁定：<span style='color:#007AFF'>{self.current_species}</span></h3>
            <p>系统已准备就绪。<br>请在右侧控制台点击 <b>[启动情感推理]</b> 开始交互。</p>
            </div>
        """)
        self.btn_gen.setEnabled(False)
        self.player_widget.setVisible(False)

    def run_llm_analysis(self):
        if not self.current_species:
            QMessageBox.warning(self, "提示", "请先在Tab1加载音频！")
            return
        self.chat_browser.append(f"<hr><b>⏳ 系统正在进行多模态情感推理...</b>")
        self.btn_llm.setEnabled(False)
        self.emo_worker = EmotionInferenceThread(self.current_audio_path, self.current_species)
        self.emo_worker.finished.connect(self.on_emotion_done)
        self.emo_worker.start()

    def on_emotion_done(self, emotion):
        if emotion in ["Unknown", "Error", "LabelErr"]:
             emotions_fallback = ['Angry', 'Happy', 'Sad', 'Alarm']
             emotion = random.choice(emotions_fallback)
             self.chat_browser.append(f"<span style='color:orange'>⚠️ 无法确定精确情绪，切换至演示模式: {emotion}</span>")
        else:
             self.chat_browser.append(f"<b>🎯 识别情绪: <span style='color:#D32F2F'>{emotion}</span></b>")
        self.chat_browser.append(f"<b>🔄 正在连接 DeepSeek 大模型获取应对策略...</b>")
        self.llm_worker = LLMWorker(self.current_species, emotion)
        self.llm_worker.finished.connect(self.on_llm_finished)
        self.llm_worker.start()

    def on_llm_finished(self, text):
        formatted = text.replace("\n", "<br>")
        self.chat_browser.append(f"""
            <div style='background:#E3F2FD; padding:15px; border-radius:10px; margin:10px 0;'>
            <b>🤖 DeepSeek助手:</b><br>{formatted}
            </div>
        """)
        match = re.search(r"(?:Prompt|提示词)[:：]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if match: self.generated_prompt = match.group(1).split("\n")[0].strip()
        else: self.generated_prompt = f"A sound of {self.current_species}"
        self.btn_llm.setEnabled(True)
        self.btn_gen.setEnabled(True)
        self.chat_browser.verticalScrollBar().setValue(self.chat_browser.verticalScrollBar().maximum())

    def run_audio_generation(self):
        if not hasattr(self, 'generated_prompt'): return
        self.chat_browser.append(f"<hr><b>🔄 正在调用 AudioLDM 生成回应音频...</b><br><span style='color:#666; font-size:12px'>Prompt: {self.generated_prompt}</span>")
        self.btn_gen.setEnabled(False)
        self.btn_gen.setText("生成中...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_sp = self.current_species.replace(" ", "_")
        filename = f"resp_{safe_sp}_{timestamp}.wav"
        save_path = BASE_DIR / "report_assets" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.gen_worker = AudioGenWorker(self.generated_prompt, str(save_path))
        self.gen_worker.finished.connect(self.on_gen_finished)
        self.gen_worker.start()

    def on_gen_finished(self, path):
        self.btn_gen.setEnabled(True)
        self.btn_gen.setText("🔊 2. 生成回应")
        if path == "error":
            self.chat_browser.append("<div style='color:red'>❌ 生成失败。</div>")
        else:
            self.chat_browser.append(f"<div style='background:#E8F5E9; padding:10px; border-radius:5px; color:#2E7D32;'><b>✅ 回应音频已生成！</b></div>")
            self.player_widget.setVisible(True)
            self.player_widget.set_media(path)
            self.player_widget.player.play()
            self.chat_browser.verticalScrollBar().setValue(self.chat_browser.verticalScrollBar().maximum())