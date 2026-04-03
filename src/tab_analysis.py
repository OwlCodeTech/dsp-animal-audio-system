# src/tab_analysis.py
import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QFileDialog, QTextBrowser, QMessageBox, QFrame, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Matplotlib 集成
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 导入公共组件 (注意这里是从 src.gui_shared 导入)
from gui_shared import BASE_DIR, STATS_CSV_PATH, MODEL_PATH, SPECIES_LABELS, AudioPlayerWidget
from preprocess import AudioCleaner
from feature_extract import FeatureAnalyzer
from train_species import SpeciesCNN
from llm_agent import AnimalLLM

# --- 线程类 (仅属于 Tab 1) ---
class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.cleaner = AudioCleaner()
        self.analyzer = FeatureAnalyzer()

    def run(self):
        try:
            segments, sr, y_trim = self.cleaner.process_single_file_memory(self.file_path, species="unknown")
            if not segments:
                self.finished.emit({"error": "无法处理音频。"})
                return
            demo_seg = segments[len(segments)//2] if len(segments) > 1 else segments[0]
            feats = self.analyzer.analyze_segment_memory(demo_seg, sr)
            
            fft_freqs = feats['fft_freq']
            fft_mags = feats['fft_mag']
            peak_idx = np.argmax(fft_mags)
            peak_freq = fft_freqs[peak_idx]
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SpeciesCNN(num_classes=len(SPECIES_LABELS)).to(device)
            pred_species = "Unknown"
            conf = 0.0
            
            if MODEL_PATH.exists():
                state = torch.load(MODEL_PATH, map_location=device)
                if state['classifier.2.weight'].shape[0] != len(SPECIES_LABELS):
                    self.finished.emit({"error": "模型类别数不匹配。"})
                    return
                model.load_state_dict(state)
                model.eval()
                inp = torch.from_numpy(feats['mfcc_feature']).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                    probs = torch.softmax(out, dim=1)
                    top_p, top_class = probs.topk(1, dim=1)
                    idx = top_class.item()
                    if 0 <= idx < len(SPECIES_LABELS):
                        pred_species = SPECIES_LABELS[idx]
                        conf = top_p.item() * 100
            
            cent = self.analyzer.compute_spectral_centroid(y_trim, sr)
            zcr = self.analyzer.compute_zcr(y_trim)
            result = {
                "y_raw": y_trim, "sr": sr, "fft_freq": feats['fft_freq'], "fft_mag": feats['fft_mag'],
                "stft_db": feats['stft_db'], "pred_species": pred_species, "confidence": conf,
                "centroid": cent, "zcr": zcr, "peak_freq": peak_freq
            }
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit({"error": str(e)})

class DSPInterpretationWorker(QThread):
    finished = pyqtSignal(str)
    def __init__(self, species, centroid, zcr, peak_freq):
        super().__init__()
        self.species = species
        self.centroid = centroid
        self.zcr = zcr
        self.peak_freq = peak_freq
        self.agent = AnimalLLM()
    def run(self):
        prompt = f"""
        你是一位资深生物声学专家。针对【{self.species}】的音频分析数据：
        1. 主频峰值 {self.peak_freq:.0f} Hz
        2. 频谱质心 {self.centroid:.0f} Hz
        3. 过零率 {self.zcr:.3f}
        
        请用通俗的语言解读这些数据反映了该动物的什么声音特征（如低沉、尖锐、清脆、浑浊）？
        请直接输出约150字的纯文本段落。
        """
        try:
            response = self.agent.client.chat.completions.create(
                model=self.agent.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            result = response.choices[0].message.content
        except Exception as e:
            result = f"AI 解读失败: {e}"
        self.finished.emit(result)

# --- Tab 1 主界面类 ---
class AnalysisTab(QWidget):
    # 定义一个信号：当分析完成时，发送 (物种名, 音频路径) 给主程序，以便传给 Tab 2
    analysis_completed = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.current_audio_path = None
        self.current_species = None
        self.current_dsp_data = None
        
        self.stats_df = None
        if STATS_CSV_PATH.exists():
            self.stats_df = pd.read_csv(STATS_CSV_PATH)
            
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 1. 左侧面板
        left_panel = QFrame()
        left_panel.setFixedWidth(180) 
        left_panel.setStyleSheet("QFrame { background: transparent; }")
        lp_layout = QVBoxLayout(left_panel)
        lp_layout.setContentsMargins(0, 0, 0, 0) 
        
        op_group = QGroupBox("操作台")
        op_layout = QVBoxLayout()
        self.btn_upload = QPushButton("📂  加载音频")
        self.btn_upload.setProperty("class", "primary") 
        self.btn_upload.setMinimumHeight(45)
        self.btn_upload.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_upload.clicked.connect(self.upload_file)
        self.lbl_file_info = QLabel("未选择文件")
        self.lbl_file_info.setAlignment(Qt.AlignCenter)
        self.lbl_file_info.setStyleSheet("color: #666; border: 1px dashed #ccc; border-radius: 5px; padding: 10px; background: #fff;")
        self.lbl_file_info.setWordWrap(True)
        self.player_widget = AudioPlayerWidget()
        op_layout.addWidget(self.btn_upload)
        op_layout.addWidget(self.lbl_file_info)
        op_layout.addSpacing(10)
        op_layout.addWidget(self.player_widget)
        op_group.setLayout(op_layout)
        
        res_group = QGroupBox("分析结果")
        res_layout = QVBoxLayout()
        self.lbl_pred_species = QLabel("...")
        self.lbl_pred_species.setAlignment(Qt.AlignCenter)
        self.lbl_pred_species.setStyleSheet("font-size: 22px; font-weight: bold; color: #333; background: #E3F2FD; border-radius: 6px; padding: 8px;")
        self.lbl_pred_conf = QLabel("置信度: -")
        self.lbl_pred_conf.setAlignment(Qt.AlignCenter)
        self.lbl_stats_cent = QLabel("质心: -")
        self.lbl_stats_zcr = QLabel("过零率: -")
        res_layout.addWidget(self.lbl_pred_species)
        res_layout.addWidget(self.lbl_pred_conf)
        res_layout.addWidget(self.lbl_stats_cent)
        res_layout.addWidget(self.lbl_stats_zcr)
        res_group.setLayout(res_layout)
        
        lp_layout.addWidget(op_group)
        lp_layout.addWidget(res_group)
        lp_layout.addStretch()
        
        # 2. 中间面板 (Matplotlib)
        center_panel = QWidget()
        cp_layout = QVBoxLayout(center_panel)
        cp_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = plt.figure(figsize=(12, 8)) 
        self.figure.patch.set_facecolor('#FFFFFF')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("border: 1px solid #ddd; border-radius: 8px;")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cp_layout.addWidget(self.canvas)
        
        # 3. 右侧面板
        right_panel = QFrame()
        right_panel.setFixedWidth(180)  
        right_panel.setStyleSheet("background: white; border-left: 1px solid #ddd;")
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(0, 0, 0, 0)
        
        ai_group = QGroupBox("🧠 专家解读")
        ai_layout = QVBoxLayout()
        self.text_dsp_report = QTextBrowser()
        self.text_dsp_report.setStyleSheet("border: none; background: #F9FAFB; padding: 10px; font-size: 13px; line-height: 1.6; color: #444;")
        self.text_dsp_report.setHtml("<div style='color:#999; text-align:center; margin-top:20px;'>暂无数据</div>")
        self.btn_interpret = QPushButton("✨ 生成报告")
        self.btn_interpret.setProperty("class", "primary")
        self.btn_interpret.clicked.connect(self.run_dsp_interpretation)
        self.btn_interpret.setEnabled(False)
        ai_layout.addWidget(self.text_dsp_report)
        ai_layout.addWidget(self.btn_interpret)
        ai_group.setLayout(ai_layout)
        rp_layout.addWidget(ai_group)

        layout.addWidget(left_panel, 0)
        layout.addWidget(center_panel, 1)
        layout.addWidget(right_panel, 0)

    def upload_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择音频', str(BASE_DIR), "Audio Files (*.wav *.mp3)")
        if fname:
            self.current_audio_path = fname
            self.lbl_file_info.setText(f"{Path(fname).name}")
            self.player_widget.set_media(fname)
            
            self.lbl_pred_species.setText("分析中...")
            self.lbl_pred_species.setStyleSheet("color: orange; font-size: 18px; font-weight: bold;")
            self.text_dsp_report.setHtml("<div style='text-align:center; padding-top:20px;'>⏳ DSP 计算进行中...</div>")
            self.btn_interpret.setEnabled(False)
            
            self.analysis_worker = AnalysisThread(fname)
            self.analysis_worker.finished.connect(self.on_analysis_done)
            self.analysis_worker.start()

    def on_analysis_done(self, data):
        if "error" in data:
            QMessageBox.critical(self, "错误", data['error'])
            self.lbl_pred_species.setText("失败")
            return
            
        self.current_species = data['pred_species']
        self.current_dsp_data = data 
        
        # 更新UI
        self.lbl_pred_species.setText(data['pred_species'])
        self.lbl_pred_species.setStyleSheet("color: #007AFF; font-size: 22px; font-weight: bold;")
        self.lbl_pred_conf.setText(f"置信度: {data['confidence']:.1f}%")
        self.lbl_stats_cent.setText(f"质心: {data['centroid']:.0f} Hz")
        self.lbl_stats_zcr.setText(f"ZCR: {data['zcr']:.3f}")
        
        # 绘图
        self.figure.clear()
        sns.set_style("whitegrid")
        gs = self.figure.add_gridspec(2, 2, hspace=0.45, wspace=0.25)

        # Waveform
        ax1 = self.figure.add_subplot(gs[0, 0])
        librosa.display.waveshow(data['y_raw'], sr=data['sr'], ax=ax1, color='#007AFF', alpha=0.6)
        ax1.set_title("Time Domain (Waveform)", fontsize=9, fontweight='bold')
        ax1.set_xlabel("Time (s)", fontsize=8)
        ax1.tick_params(axis='both', which='major', labelsize=7)
        
        # FFT
        ax2 = self.figure.add_subplot(gs[0, 1])
        n = len(data['fft_freq']) // 2
        ax2.plot(data['fft_freq'][:n], data['fft_mag'][:n], color='#FF9500', linewidth=1)
        ax2.set_title("Frequency Domain (FFT)", fontsize=9, fontweight='bold')
        ax2.set_xlabel("Frequency (Hz)", fontsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=7)
        ax2.grid(True, alpha=0.2)
        
        # STFT
        ax3 = self.figure.add_subplot(gs[1, 0])
        img = librosa.display.specshow(data['stft_db'], sr=data['sr'], x_axis='time', y_axis='log', ax=ax3, cmap='magma')
        ax3.set_title("Spectrogram (STFT)", fontsize=9, fontweight='bold')
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.tick_params(axis='both', which='major', labelsize=7)
        self.figure.colorbar(img, ax=ax3, format="%+2.0f dB", shrink=0.8)
        
        # Boxplot
        ax4 = self.figure.add_subplot(gs[1, 1])
        if self.stats_df is not None:
            sns.boxplot(x="Species", y="Centroid_Mean", data=self.stats_df, ax=ax4, palette="Pastel1", linewidth=0.8, fliersize=2)
            try:
                species_list = sorted(self.stats_df['Species'].unique())
                if self.current_species in species_list:
                    x_idx = species_list.index(self.current_species)
                    ax4.scatter(x_idx, data['centroid'], color='red', s=80, zorder=10, marker='D', edgecolors='white', label='Current')
            except: pass
            ax4.set_title("Stats (Spectral Centroid)", fontsize=9, fontweight='bold')
            ax4.set_xlabel("")
            ax4.tick_params(axis='x', rotation=30, labelsize=7)
            ax4.tick_params(axis='y', labelsize=7)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        self.btn_interpret.setEnabled(True)
        self.text_dsp_report.setHtml("<div style='padding:10px;'>✅ <b>图表绘制完毕。</b><br>点击下方按钮，让DeepSeek为您解读这些声学特征的生物学含义。</div>")
        
        # 发送信号给 Tab 2
        self.analysis_completed.emit(self.current_species, self.current_audio_path)

    def run_dsp_interpretation(self):
        self.text_dsp_report.setHtml("<div style='color:#007AFF; text-align:center;'>🧠 DeepSeek正在分析频谱数据...</div>")
        self.btn_interpret.setEnabled(False)
        worker = DSPInterpretationWorker(
            self.current_species, 
            self.current_dsp_data['centroid'], 
            self.current_dsp_data['zcr'],
            self.current_dsp_data['peak_freq']
        )
        worker.finished.connect(self.on_dsp_interpret_done)
        worker.start()
        self.dsp_worker = worker

    def on_dsp_interpret_done(self, text):
        self.text_dsp_report.setHtml(f"<div style='font-family:sans-serif; line-height:1.5; color:#333;'>{text}</div>")
        self.btn_interpret.setEnabled(True)