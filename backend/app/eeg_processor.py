"""
EEG Signal Processing Module
脑电信号处理核心模块
"""

import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EEGProcessor:
    """
    EEG 信号处理器
    
    功能：
    - 信号导入（EDF/BDF/CSV）
    - 预处理（滤波、去噪、伪迹去除）
    - 特征提取（功率谱、频段功率、相干性）
    - AI 分析（专注度、情绪识别）
    """
    
    # EEG 频段定义 (Hz)
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    }
    
    def __init__(self, sample_rate: int = 256, n_channels: int = 14):
        """
        初始化 EEG 处理器
        
        Args:
            sample_rate: 采样率 (Hz)
            n_channels: 通道数
        """
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.filters = self._init_filters()
        
        logger.info(f"EEGProcessor initialized: {sample_rate}Hz, {n_channels} channels")
    
    def _init_filters(self) -> Dict:
        """初始化滤波器"""
        filters = {}
        
        # 带通滤波器 (0.5-45Hz)
        sos_bp = signal.butter(4, [0.5, 45], btype='band', 
                               fs=self.sample_rate, output='sos')
        filters['bandpass'] = sos_bp
        
        # 工频陷波滤波器 (50Hz)
        sos_notch = signal.iirnotch(50, 30, self.sample_rate)
        filters['notch'] = sos_notch
        
        return filters
    
    def load_edf(self, filepath: str) -> np.ndarray:
        """
        加载 EDF 文件
        
        Args:
            filepath: EDF 文件路径
            
        Returns:
            EEG 数据 (n_channels, n_samples)
        """
        try:
            import pyedflib
            f = pyedflib.EdfReader(filepath)
            n_channels = f.signals_in_file
            signal_labels = f.getSignalLabels()
            
            data = np.zeros((n_channels, f.getNSamples()[0]))
            for i in range(n_channels):
                data[i, :] = f.readSignal(i)
            
            f.close()
            logger.info(f"Loaded EDF file: {filepath}, shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Failed to load EDF file: {e}")
            raise
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        预处理 EEG 信号
        
        步骤：
        1. 带通滤波 (0.5-45Hz)
        2. 工频滤波 (50Hz 陷波)
        3. 伪迹去除
        
        Args:
            data: 原始 EEG 数据 (n_channels, n_samples)
            
        Returns:
            预处理后的数据
        """
        # 带通滤波
        filtered = signal.sosfiltfilt(self.filters['bandpass'], data, axis=-1)
        
        # 工频滤波
        cleaned = signal.sosfiltfilt(self.filters['notch'][0], filtered, axis=-1)
        
        # 伪迹去除（简单阈值法）
        cleaned = self._remove_artifacts(cleaned)
        
        logger.info(f"Preprocessing complete: {data.shape} -> {cleaned.shape}")
        return cleaned
    
    def _remove_artifacts(self, data: np.ndarray, threshold: float = 100.0) -> np.ndarray:
        """
        去除伪迹（眼动、肌电等）
        
        Args:
            data: EEG 数据
            threshold: 伪迹阈值 (μV)
            
        Returns:
            去伪迹后的数据
        """
        cleaned = data.copy()
        
        # 检测并插值异常值
        for i in range(data.shape[0]):
            mask = np.abs(data[i]) > threshold
            if np.any(mask):
                # 线性插值替换异常值
                cleaned[i, mask] = np.interp(
                    np.where(mask)[0],
                    np.where(~mask)[0],
                    data[i, ~mask]
                )
        
        return cleaned
    
    def extract_features(self, data: np.ndarray) -> Dict:
        """
        提取 EEG 特征
        
        Args:
            data: 预处理后的 EEG 数据
            
        Returns:
            特征字典
        """
        features = {
            'power_spectrum': self._calc_power_spectrum(data),
            'band_power': self._calc_band_power(data),
            'coherence': self._calc_coherence(data),
            'asymmetry': self._calc_asymmetry(data),
        }
        
        logger.info(f"Features extracted: {list(features.keys())}")
        return features
    
    def _calc_power_spectrum(self, data: np.ndarray) -> np.ndarray:
        """计算功率谱密度"""
        freqs, psd = signal.welch(
            data, 
            self.sample_rate, 
            nperseg=self.sample_rate * 2,
            axis=-1
        )
        return psd
    
    def _calc_band_power(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算各频段功率
        
        Returns:
            各频段功率字典
        """
        psd = self._calc_power_spectrum(data)
        freqs = np.fft.rfftfreq(data.shape[-1], 1/self.sample_rate)
        
        band_power = {}
        for band, (low, high) in self.BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            band_power[band] = np.mean(psd[:, mask], axis=-1)
        
        return band_power
    
    def _calc_coherence(self, data: np.ndarray) -> np.ndarray:
        """计算通道间相干性"""
        n_channels = data.shape[0]
        coherence = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                f, coh = signal.coherence(
                    data[i], data[j], 
                    self.sample_rate,
                    nperseg=self.sample_rate
                )
                coherence[i, j] = np.mean(coh)
                coherence[j, i] = coherence[i, j]
        
        return coherence
    
    def _calc_asymmetry(self, data: np.ndarray) -> Dict[str, float]:
        """
        计算左右脑不对称性（前额叶）
        
        Returns:
            各频段不对称性
        """
        band_power = self._calc_band_power(data)
        
        # 假设通道 0-6 为左脑，7-13 为右脑（根据实际设备调整）
        left_idx = range(0, 7)
        right_idx = range(7, 14)
        
        asymmetry = {}
        for band, power in band_power.items():
            left_power = np.mean(power[left_idx])
            right_power = np.mean(power[right_idx])
            asymmetry[band] = np.log10(right_power) - np.log10(left_power)
        
        return asymmetry
    
    def analyze(self, data: np.ndarray) -> Dict:
        """
        AI 分析：专注度、情绪等
        
        Args:
            data: EEG 数据
            
        Returns:
            分析结果
        """
        features = self.extract_features(data)
        
        # 简化的专注度计算（实际应使用训练好的模型）
        alpha = features['band_power']['alpha']
        beta = features['band_power']['beta']
        theta = features['band_power']['theta']
        
        # 专注度 = beta / (alpha + theta)
        attention_score = np.mean(beta) / (np.mean(alpha) + np.mean(theta) + 1e-6)
        attention_score = np.clip(attention_score * 10, 0, 100)
        
        # 放松度 = alpha / beta
        relaxation_score = np.mean(alpha) / (np.mean(beta) + 1e-6)
        relaxation_score = np.clip(relaxation_score * 10, 0, 100)
        
        # 情绪 valence（基于左右脑不对称性）
        valence = np.tanh(features['asymmetry']['alpha'])
        
        # 情绪 arousal（基于 beta/alpha 比率）
        arousal = np.mean(beta) / (np.mean(alpha) + 1e-6)
        arousal = np.clip(arousal / 5, 0, 1)
        
        return {
            'attention_score': float(attention_score),
            'relaxation_score': float(relaxation_score),
            'valence': float(valence),
            'arousal': float(arousal),
            'band_power': {k: v.tolist() for k, v in features['band_power'].items()},
        }


# 便捷函数
def process_eeg_file(filepath: str, sample_rate: int = 256) -> Dict:
    """
    处理 EEG 文件的便捷函数
    
    Args:
        filepath: EDF 文件路径
        sample_rate: 采样率
        
    Returns:
        分析结果
    """
    processor = EEGProcessor(sample_rate=sample_rate)
    data = processor.load_edf(filepath)
    cleaned = processor.preprocess(data)
    result = processor.analyze(cleaned)
    return result


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        result = process_eeg_file(sys.argv[1])
        print(f"Attention: {result['attention_score']:.1f}")
        print(f"Relaxation: {result['relaxation_score']:.1f}")
        print(f"Valence: {result['valence']:.2f}")
        print(f"Arousal: {result['arousal']:.2f}")
