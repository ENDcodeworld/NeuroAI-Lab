"""脑电信号预处理工具"""

import numpy as np
from scipy import signal


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """带通滤波"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def notch_filter(data, freq, fs, Q=30):
    """陷波滤波（去除工频干扰）"""
    b, a = signal.iirnotch(freq, Q, fs)
    return signal.filtfilt(b, a, data, axis=-1)


def extract_features(eeg_data, fs=256):
    """提取 EEG 特征"""
    features = {}
    
    # 频段划分
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    for band_name, (low, high) in bands.items():
        filtered = bandpass_filter(eeg_data, low, high, fs)
        # 计算功率谱密度
        freqs, psd = signal.welch(filtered, fs, nperseg=min(256, fs * 2))
        mask = (freqs >= low) & (freqs <= high)
        features[f'{band_name}_power'] = np.mean(psd[mask])
        features[f'{band_name}_peak'] = freqs[mask][np.argmax(psd[mask])]
    
    return features


if __name__ == "__main__":
    # 模拟 EEG 数据
    fs = 256
    duration = 5  # seconds
    t = np.linspace(0, duration, fs * duration)
    
    # 生成模拟信号
    eeg = (np.sin(2 * np.pi * 10 * t) +   # Alpha
           0.5 * np.sin(2 * np.pi * 20 * t) +  # Beta
           0.1 * np.random.randn(len(t)))    # Noise
    
    features = extract_features(eeg, fs)
    print("EEG Features:")
    for k, v in features.items():
        print(f"  {k}: {v:.6f}")
