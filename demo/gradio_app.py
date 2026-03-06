"""
NeuroAI-Lab Gradio Demo
EEG 信号分析与专注力检测演示

运行方式:
    python app.py
    
部署到 HuggingFace Spaces:
    1. 创建 Space: https://huggingface.co/spaces
    2. 选择 Gradio 作为 SDK
    3. 上传此文件和 requirements.txt
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_eeg_signal(duration=10, sample_rate=256, focus_level=0.5):
    """
    生成模拟 EEG 信号
    
    参数:
        duration: 信号时长 (秒)
        sample_rate: 采样率 (Hz)
        focus_level: 专注度 (0-1)，越高 Beta 波越强
    """
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # 各频段脑电波
    delta = 0.5 * np.sin(2 * np.pi * 2 * t)  # 0.5-4 Hz: 深度睡眠
    theta = 0.3 * np.sin(2 * np.pi * 6 * t)  # 4-8 Hz: 放松/困倦
    alpha = 0.4 * np.sin(2 * np.pi * 10 * t)  # 8-13 Hz: 放松清醒
    beta = (0.2 + 0.3 * focus_level) * np.sin(2 * np.pi * 20 * t)  # 13-30 Hz: 专注
    gamma = 0.15 * np.sin(2 * np.pi * 40 * t)  # 30-100 Hz: 高度认知
    
    # 合成信号
    eeg = delta + theta + alpha + beta + gamma
    
    # 添加噪声
    noise = np.random.normal(0, 0.1, len(eeg))
    eeg = eeg + noise
    
    return t, eeg

def analyze_eeg(eeg_data, sample_rate=256):
    """
    分析 EEG 信号的频段功率
    """
    # 使用 Welch 方法计算功率谱密度
    freqs, psd = signal.welch(eeg_data, sample_rate, nperseg=512)
    
    # 计算各频段功率
    def band_power(freq_range):
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        return np.trapz(psd[mask], freqs[mask])
    
    delta_power = band_power([0.5, 4])
    theta_power = band_power([4, 8])
    alpha_power = band_power([8, 13])
    beta_power = band_power([13, 30])
    gamma_power = band_power([30, 100])
    
    total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
    
    # 计算百分比
    percentages = {
        'Delta': delta_power / total_power * 100,
        'Theta': theta_power / total_power * 100,
        'Alpha': alpha_power / total_power * 100,
        'Beta': beta_power / total_power * 100,
        'Gamma': gamma_power / total_power * 100
    }
    
    # 计算专注度指数 (Beta/Alpha 比率)
    focus_index = beta_power / (alpha_power + 1e-6)
    
    # 计算放松度指数 (Alpha/Theta 比率)
    relaxation_index = alpha_power / (theta_power + 1e-6)
    
    return percentages, focus_index, relaxation_index, freqs, psd

def simulate_and_analyze(focus_level, duration):
    """
    模拟 EEG 信号并分析
    """
    # 生成信号
    t, eeg = generate_eeg_signal(duration, focus_level=focus_level)
    
    # 分析信号
    percentages, focus_index, relaxation_index, freqs, psd = analyze_eeg(eeg)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图 1: 时域信号
    ax1 = axes[0, 0]
    ax1.plot(t[:1000], eeg[:1000], 'b-', linewidth=0.5)  # 只显示前 4 秒
    ax1.set_xlabel('时间 (秒)', fontsize=11)
    ax1.set_ylabel('振幅 (μV)', fontsize=11)
    ax1.set_title('EEG 时域信号 (前 4 秒)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 图 2: 功率谱密度
    ax2 = axes[0, 1]
    ax2.semilogy(freqs, psd, 'r-', linewidth=1)
    ax2.set_xlabel('频率 (Hz)', fontsize=11)
    ax2.set_ylabel('功率谱密度', fontsize=11)
    ax2.set_title('EEG 功率谱密度', fontsize=12, fontweight='bold')
    ax2.axvline(x=4, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=8, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=13, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=30, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 图 3: 频段功率分布
    ax3 = axes[1, 0]
    bands = list(percentages.keys())
    powers = list(percentages.values())
    colors = ['#663399', '#4477AA', '#44AA44', '#DDAA44', '#DD4444']
    bars = ax3.bar(bands, powers, color=colors, alpha=0.8)
    ax3.set_ylabel('功率百分比 (%)', fontsize=11)
    ax3.set_title('各频段功率分布', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    
    # 在柱状图上添加数值
    for bar, power in zip(bars, powers):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{power:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 图 4: 专注度和放松度
    ax4 = axes[1, 1]
    metrics = ['专注度指数\n(Beta/Alpha)', '放松度指数\n(Alpha/Theta)']
    values = [focus_index, relaxation_index]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
    ax4.set_ylabel('指数值', fontsize=11)
    ax4.set_title('神经反馈指标', fontsize=12, fontweight='bold')
    
    # 添加数值
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # 生成状态描述
    if focus_index > 1.5:
        status = "🎯 高度专注状态 - Beta 波活跃，适合深度工作"
    elif focus_index > 0.8:
        status = "📚 正常专注状态 - 适合学习和阅读"
    elif focus_index > 0.5:
        status = "😌 放松状态 - Alpha 波主导，适合休息"
    else:
        status = "😴 疲劳状态 - Theta/Delta 波增加，需要休息"
    
    return fig, status, percentages

# 创建 Gradio 界面
with gr.Blocks(title="NeuroAI-Lab Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 NeuroAI-Lab EEG 信号分析 Demo
    
    体验脑电波 (EEG) 信号分析与专注力检测！
    
    **使用说明**：
    1. 调整专注度水平和信号时长
    2. 点击"生成并分析 EEG 信号"
    3. 查看时域波形、功率谱、频段分布和神经反馈指标
    
    🔗 完整项目：[GitHub](https://github.com/ENDcodeworld/NeuroAI-Lab)
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔧 参数设置")
            focus_slider = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=0.5, 
                step=0.05,
                label="专注度水平 (Focus Level)"
            )
            duration_slider = gr.Slider(
                minimum=5, 
                maximum=30, 
                value=10, 
                step=5,
                label="信号时长 (秒)"
            )
            analyze_btn = gr.Button("🔍 生成并分析 EEG 信号", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 📊 分析结果")
            status_text = gr.Textbox(label="状态评估", lines=2)
    
    output_plot = gr.Plot(label="EEG 分析图表")
    
    # 绑定事件
    analyze_btn.click(
        fn=simulate_and_analyze,
        inputs=[focus_slider, duration_slider],
        outputs=[output_plot, status_text]
    )
    
    gr.Markdown("""
    ---
    **关于 NeuroAI-Lab**
    
    NeuroAI-Lab 专注于脑机接口 (BCI) 与人工智能的融合创新：
    - 🔬 EEG 数据分析工具
    - 🎯 神经反馈训练 App
    - 📊 BCI 技术追踪平台
    
    由 [ENDcodeworld](https://github.com/ENDcodeworld) 开发 | MIT License
    """)

if __name__ == "__main__":
    demo.launch()
