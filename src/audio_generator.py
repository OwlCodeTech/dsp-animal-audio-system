import torch
import scipy.io.wavfile
import numpy as np  # 新增：用于数值处理
import os
from diffusers import AudioLDMPipeline, DPMSolverMultistepScheduler
from pathlib import Path

# 自动定位本地模型路径
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = BASE_DIR / "models" / "audioldm-s-full-v2"

class AudioGenerator:
    # 类变量存储模型，实现“单例模式”
    _pipeline_cache = None 

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- 提示词库 ---
        self.species_prompts = {
    'monkey': {
        'alarm': """
            Primate alarm call signaling immediate danger. Sharp, loud, high-pitched shrieks repeated in rapid succession.
            Acoustic features: Frequency range 2-4kHz, short bursts (0.2-0.5 sec), intense energy in upper harmonics, abrupt onset and offset.
            Context: Rainforest canopy, monkey troop spotting a predator (leopard or eagle). Multiple individuals joining in alarm chorus.
            Reference: Vervet monkey or capuchin monkey alarm calls with distinct predator-specific variations.
            Audio quality: Natural forest ambience with layered calls from different distances.
            """,
        
        'call': """
            Primate social contact and communication calls. Varied vocalizations for group cohesion, ranging from soft coos to moderate chirps.
            Acoustic features: Mixed frequencies (500Hz-3kHz), melodic contours, conversational turn-taking patterns among troop members.
            Context: Daily foraging activity in tropical forest, maintaining group cohesion while moving through trees.
            Reference: Howler monkey long-distance calls or marmoset social chatter, depending on species.
            Audio quality: Clear recording capturing both individual calls and group dynamics.
            """,
        
        'angry': """
            Primate aggressive display and threat vocalization. Deep, guttural growls escalating to loud, harsh screams.
            Acoustic features: Low-frequency components (100-300Hz) combined with high-frequency screeches (2-5kHz), amplitude building to climax.
            Context: Territorial dispute between two male monkeys, displaying teeth, shaking branches.
            Reference: Baboon aggressive confrontation sounds or macaque dominance challenge vocalizations.
            Audio quality: Close perspective capturing intensity and physicality of confrontation.
            """,
        
        'worry': """
            Primate anxious or uncertain vocalizations. Soft, repetitive whimpers or uneasy grunts expressing discomfort or mild distress.
            Acoustic features: Lower volume than alarm calls, irregular rhythm, frequency modulation suggesting unease (800Hz-1.5kHz).
            Context: Unfamiliar situation or novel object causing cautious concern, troop members exchanging nervous sounds.
            Reference: Chimpanzee or bonobo uncertain vocalizations when facing ambiguous threats.
            Audio quality: Intimate recording capturing subtle emotional state through vocal nuances.
            """
    },
    
    'dog': {
        'angry': """
            Canine aggressive threat with full emotional intensity. Deep, sustained growl escalating to explosive barks.
            Acoustic features: Fundamental frequency 70-180Hz, chest resonance, sharp bark attacks at 2-3 second intervals.
            Context: Home territory defense, dog confronting intruder at property boundary. Body tense, teeth bared.
            Reference: Protective breed (Doberman, German Shepherd) territorial defense sequence.
            Audio quality: Powerful, dynamic recording capturing both low growls and sharp barks.
            """,
        
        'happy': """
            Canine joyful expression and contentment. Playful barks, panting, and excited whimpers during positive interaction.
            Acoustic features: Bright tonal quality, rising pitch contours, rapid panting (2-4 breaths/second), tail-wagging sounds.
            Context: Owner returning home, dog expressing enthusiastic greeting. Play bow invitation, toy in mouth.
            Reference: Labrador or Golden Retriever exuberant greeting behavior with full body wagging.
            Audio quality: Warm, close recording capturing breath sounds and movement rustling.
            """,
        
        'sad': """
            Canine distress or loneliness vocalization. Whining, soft howling, or subdued vocal expressions of sadness.
            Acoustic features: High-pitched whines (800Hz-2kHz), slow tempo, downward pitch contours, intermittent sighing.
            Context: Separation anxiety, dog alone at home. Waiting by door, occasional glances toward exit.
            Reference: Rescue dog vocalizations or puppy distress calls when separated from littermates.
            Audio quality: Intimate, emotional capture conveying longing and melancholy.
            """
    },
    
    'deer': {
        'alarm': """
            Cervid acute danger alert system. Loud, explosive snort followed by stamping foot sounds and retreat.
            Acoustic features: Sharp exhalation (0.3-0.8 sec) with broad frequency spectrum (500Hz-4kHz), carrying through forest.
            Context: Edge of woodland meadow, deer detecting human or predator scent. Immediate freeze then alarm.
            Reference: White-tailed deer classic alarm snort, often followed by flag-tail display and bounding escape.
            Audio quality: Crisp outdoor recording with natural reverb and spatial awareness.
            """,
        
        'mating_call': """
            Cervid rutting season reproductive vocalizations. Deep, guttural groans of stag and responsive calls of does.
            Acoustic features: Stag: low-frequency grunts (80-250Hz) in series; Doe: higher-pitched mews (1-1.8kHz).
            Context: Autumn breeding ground, dominant stag gathering and guarding harem. Challenges from younger males.
            Reference: Red deer or elk rutting calls, iconic sounds of seasonal mating rituals.
            Audio quality: Atmospheric recording capturing both close calls and distant echoes in valley.
            """,
        
        'song': """
            Artistic interpretation of deer vocalization as melodic expression. Sustained, tone-like calls arranged rhythmically.
            Acoustic features: Modified natural calls (based on contact or mating sounds) organized into repeating melodic phrases.
            Context: Mythical forest clearing at dawn, deer emitting haunting, musical calls that seem intentionally patterned.
            Reference: Fantasy film soundtrack where animal sounds are orchestrated into musical elements.
            Audio quality: Cinematic processing with subtle reverb, EQ shaping for musicality.
            """
    },
    
    'Gallus gallus': {
        'alarm': """
            Galliform avian predator alert system. Rapid, sharp cackling with distinct urgency and arousal.
            Acoustic features: High-intensity clucking (2-3.5kHz) in rapid bursts, alarm-specific rhythm different from日常 calls.
            Context: Free-range chicken flock spotting aerial predator (hawk, eagle). Immediate cessation of foraging, heads up.
            Reference: Chicken aerial predator alarm call (distinct from ground predator alarm).
            Audio quality: Clear field recording capturing the immediate shift from calm to alarm state.
            """,
        
        'call': """
            Domestic fowl日常 communication and social vocalizations. Wide variety of clucks, purrs, and chirps for flock coordination.
            Acoustic features: Frequency range 300Hz-2kHz, rhythmic patterns conveying specific information (food discovery, location, etc.).
            Context: Daytime foraging in barnyard, hens communicating with chicks and other flock members.
            Reference: Classic chicken vocabulary including food calls, contentment purrs, and location notes.
            Audio quality: Naturalistic recording of active flock with spatial distribution of individuals.
            """,
        
        'song': """
            Artistic rooster crow arrangement and variation. Multiple crows arranged rhythmically or melodically.
            Acoustic features: Rooster crow fundamentals (500-700Hz) with variations in timing, pitch modulation, and harmony.
            Context: Sunrise chorus of multiple roosters in rural landscape, creating natural polyphony.
            Reference: Traditional "cock-a-doodle-doo" but arranged as intentional musical performance.
            Audio quality: Dawn atmosphere with spatial depth, multiple distances, natural reverb.
            """
    }
}

        # 如果缓存中已有模型，直接复用
        if AudioGenerator._pipeline_cache is not None:
            self.pipe = AudioGenerator._pipeline_cache
            self.model_loaded = True
            return

        print(f"正在初始化 AudioLDM (Device: {self.device})...")
        
        try:
            # 优先加载本地模型，不存在则从 HuggingFace 下载
            model_path = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "cvssp/audioldm-s-full-v2"
            
            if LOCAL_MODEL_PATH.exists():
                print(f"Loading from local: {model_path}")
            else:
                print("Local model not found, downloading from HuggingFace...")

            # 加载管线
            self.pipe = AudioLDMPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            # 切换为 DPM Solver 调度器（加速推理）
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # 显存优化
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()

            # 存入缓存
            AudioGenerator._pipeline_cache = self.pipe
            self.model_loaded = True
            print("✅ AudioLDM 加载完成 (加速模式)")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model_loaded = False

    def get_species_prompt(self, species, emotion):
        """获取物种特定的提示词"""
        species = species.lower()
        emotion = emotion.lower()
        
        if species not in self.species_prompts:
            print(f"⚠️ 不支持的物种: {species}，使用通用生成")
            return f"{species} making {emotion} sounds"
        
        if emotion not in self.species_prompts[species]:
            print(f"⚠️ 不支持的 {species} 情绪: {emotion}，使用通用生成")
            return f"{species} making {emotion} sounds"
        
        return self.species_prompts[species][emotion]

    def generate(self, prompt, save_path):
        """
        生成音频的核心函数 (核心逻辑已替换为更稳健的 NumPy 处理版本)
        """
        if not self.model_loaded:
            print("Error: Model not loaded.")
            return False
            
        print(f"Generating (10s): {prompt}")
        
        try:
            # 1. 模型推理
            # 注意：移除了强制的正向/负向提示词，完全依赖输入
            result = self.pipe(
                prompt, 
                num_inference_steps=15,    # 跟随新代码：使用15步，速度更快
                audio_length_in_s=10.0,    # 保持原来的10秒时长需求
                guidance_scale=3.0         # 跟随新代码：降低引导系数，听感更自然
            )
            
            # 2. 安全获取音频数据
            if hasattr(result, 'audios') and len(result.audios) > 0:
                audio = result.audios[0]
            else:
                print("❌ 无法获取生成的音频数据")
                return False

            # 3. 数据类型转换 (Tensor -> NumPy)
            # 这是为了防止某些环境直接存 Tensor 报错
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            
            # 确保是 float32 类型
            audio = audio.astype(np.float32)

            # 4. 音量归一化 (关键改进)
            # 避免声音太小或太大导致的削波
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.8  # 归一化到 0.8 幅度
            else:
                print("⚠️ 生成的音频全为静音")

            # 5. 保存文件
            scipy.io.wavfile.write(save_path, 16000, audio)
            print(f"✅ 音频已保存: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ 生成过程中出错: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误栈，方便调试
            return False

    def generate_from_species_emotion(self, species, emotion, save_path):
        """
        根据物种和情绪生成音频的便捷接口
        """
        base_prompt = self.get_species_prompt(species, emotion)
        
        print(f"🎯 正在处理: {species} - {emotion}")
        print(f"📝 使用提示词: {base_prompt}")
        
        return self.generate(base_prompt, save_path)

# 测试代码
if __name__ == "__main__":
    generator = AudioGenerator()
    
    output_dir = Path("generated_samples")
    output_dir.mkdir(exist_ok=True)
    
    print("\n--- 测试融合功能 (使用新的数据处理逻辑) ---")
    test_cases = [
        ('dog', 'angry'),
        ('cat', 'happy'),
        ('deer', 'alarm')
    ]
    
    for species, emotion in test_cases:
        filename = output_dir / f"test_{species}_{emotion}.wav"
        generator.generate_from_species_emotion(species, emotion, str(filename))