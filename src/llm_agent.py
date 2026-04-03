import os
from openai import OpenAI

# ================= Deepseek API 配置区域 =================
DEEPSEEK_API_KEY = "sk-c7e83b84f9184096a838f7e0f5775476"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
# ==========================================================

class AnimalLLM:
    def __init__(self):
        try:
            self.client = OpenAI(
                base_url=DEEPSEEK_BASE_URL,
                api_key=DEEPSEEK_API_KEY
            )
            self.model_id = DEEPSEEK_MODEL
            self.is_ready = True
            print(f"✅ Deepseek API 连接初始化: {self.model_id}")
        except Exception as e:
            print(f"❌ Deepseek API 初始化失败: {e}")
            self.is_ready = False

    def analyze(self, species, emotion):
        if not self.is_ready:
            return "❌ 配置错误: 请检查 src/llm_agent.py"

        # 根据物种获取具体声学参数
        acoustic_params = self._get_species_params(species)
        
        prompt = f"""
        你是一个精通动物行为学、生物声学和音频工程的专家。我的声学系统检测到一只【{species}】正处于【{emotion}】情绪状态。

        ### 任务一：中文分析与策略
        1. **行为分析**（80-120字）：
           - 解释这种情绪的生物学背景和行为意义
           - 分析该物种在这种情绪下的典型表现
           - 描述相关的发声特征和声学模式
        2. **交流策略**（60-100字）：
           - 提供详细的人类应对方案
           - 建议建立声学交流的具体方法
           - 包括必要的安全注意事项

        ### 任务二：英文音频生成提示词 (CRITICAL - 避免电音的关键)
        创建一个用于音频生成的详细提示词，用于生成一只{species}表达{emotion}情绪的真实叫声。
        
        **⚠️ 消除电音的关键要求：**
        1. **音频来源要求**:
           - STRICTLY use only authentic, high-quality FIELD RECORDINGS
           - ABSOLUTELY NO synthetic voices, NO text-to-speech, NO vocoders
           - Source material must be unprocessed wildlife recordings
           
        2. **技术参数要求** (针对{species}):
           {acoustic_params}
           
        3. **自然度增强**:
           - Include subtle mouth movement sounds (tongue, lips, breath)
           - Add natural reverb from the animal's environment
           - Preserve all harmonic overtones and natural frequency modulation
           - Include slight variations in timing and pitch (avoid robotic repetition)
           - Add subtle background ambience from natural habitat
           
        4. **禁止的人工痕迹**:
           - NO electronic buzzing or humming
           - NO clipping or digital distortion
           - NO autotune or pitch correction artifacts
           - NO artificial reverb or delay effects
           - NO normalized/compressed dynamics
           
        5. **情感表达细节**:
           - The {emotion} should be conveyed through natural vocal characteristics
           - Include appropriate emotional micro-variations (slight tremors, breath changes)
           - Response should be contextually appropriate for social communication
           
        6. **最终提示词结构**:
           - "Generate a completely natural field recording of a {species} expressing {emotion}"
           - "Use only authentic wildlife recordings with zero processing"
           - "Emphasis on organic, unprocessed natural sounds"
           - "Duration: 10-15 seconds with natural fade in/out"

        ### 输出格式：
        【行为分析】
        [你的详细中文分析]

        【交流策略】
        [你的中文策略建议]

        【Audio_Prompt】
        [你的完整英文提示词]
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "你是动物声学专家，擅长生成避免电音的自然声音提示词。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Deepseek API 调用失败: {str(e)}"
    
    def _get_species_params(self, species):
        """
        获取物种特定的声学参数
        更新说明：已根据新目录结构更新物种列表 (monkey, dog, deer, gallus gallus)
        """
        params_db = {
            "monkey": {
                "frequency_range": "200-6000 Hz (complex harmonic structure)",
                "sample_rate": "96 kHz minimum",
                "bit_depth": "24-bit",
                "key_characteristics": "Rapid temporal modulation, chatter, screams, formant variations"
            },
            "dog": {
                "frequency_range": "150-1000 Hz (fundamental), harmonics up to 8000 Hz",
                "sample_rate": "96 kHz minimum",
                "bit_depth": "24-bit",
                "key_characteristics": "Rough vocal folds, wide frequency modulation, panting sounds"
            },
            "deer": {
                "frequency_range": "100-3000 Hz (low frequency grunts/bellows)",
                "sample_rate": "96 kHz",
                "bit_depth": "24-bit",
                "key_characteristics": "Vocal tract resonance, guttural sounds, nasal tones"
            },
            "Gallus gallus": {
                "frequency_range": "250-5000 Hz (clucks and crows)",
                "sample_rate": "192 kHz for sharp transients",
                "bit_depth": "32-bit float",
                "key_characteristics": "Sharp attack transients, rapid amplitude modulation, crowing harmonics"
            }
        }
        
        # 为了兼容性，处理大小写或未匹配的情况
        # 尝试直接匹配，如果不行则尝试小写匹配
        key = species
        if species not in params_db:
             # 简单的查找逻辑
             for k in params_db:
                 if k.lower() == species.lower():
                     key = k
                     break

        # 获取参数或默认值
        params = params_db.get(
            key, 
            {
                "frequency_range": "200-8000 Hz (general biological range)",
                "sample_rate": "96 kHz",
                "bit_depth": "24-bit",
                "key_characteristics": "Natural vocal fold vibrations, environmental resonance"
            }
        )
        
        return f"""
        - Frequency range: {params['frequency_range']}
        - Required sample rate: {params['sample_rate']}
        - Bit depth: {params['bit_depth']}
        - Key acoustic characteristics: {params['key_characteristics']}
        """

# 测试代码
if __name__ == "__main__":
    agent = AnimalLLM()
    print("正在测试连接...")
    
    # 测试案例：更新为新目录结构中的物种和情绪
    test_cases = [
        ("monkey", "alarm"),      # 对应 monkey/alarm
        ("deer", "mating_call"),  # 对应 deer/mating_call
        ("Gallus gallus", "song") # 对应 Gallus gallus/song
    ]
    
    for species, emotion in test_cases:
        print(f"\n--- 测试案例：{species} / {emotion} ---")
        result = agent.analyze(species, emotion)
        # 打印部分结果以验证
        if result and len(result) > 100:
             print(result[:500] + "...")
        else:
             print(result)