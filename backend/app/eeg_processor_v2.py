"""
Advanced EEG Signal Processing Module
脑电信号处理核心模块 - 增强版

功能:
- 信号导入（EDF/BDF/CSV/MNE 支持格式）
- 预处理（滤波、去噪、伪迹去除）
- 高级伪迹去除（ICA、ASR）
- 特征提取（功率谱、频段功率、相干性、微状态）
- 时频分析（小波变换、STFT）
- AI 分析（专注度、情绪识别）

作者：NeuroAI-Lab Team
版本：2.0.0
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Optional, Tuple, List, Union
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class BandType(Enum):
    """EEG 频段定义"""
    DELTA = 'delta'      # 0.5-4 Hz: 深度睡眠
    THETA = 'theta'      # 4-8 Hz: 放松、创造力
    ALPHA = 'alpha'      # 8-13 Hz: 放松清醒
    BETA = 'beta'        # 13-30 Hz: 专注、思考
    GAMMA = 'gamma'      # 30-45 Hz: 高度认知


@dataclass
class EEGConfig:
    """EEG 处理器配置"""
    sample_rate: int = 256
    n_channels: int = 14
    filter_order: int = 4
    notch_freq: float = 50.0
    notch_q: float = 30.0
    artifact_threshold: float = 100.0  # μV
    
    # 频段定义
    bands: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.bands is None:
            self.bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45),
            }


class EEGProcessor:
    """
    高级 EEG 信号处理器
    
    功能模块:
    1. 信号导入与导出
    2. 预处理（滤波、去噪）
    3. 伪迹去除（ICA、ASR）
    4. 特征提取
    5. 时频分析
    6. AI 分析
    """
    
    def __init__(self, config: Optional[EEGConfig] = None):
        """
        初始化 EEG 处理器
        
        Args:
            config: 处理器配置，默认使用标准配置
        """
        self.config = config or EEGConfig()
        self.sample_rate = self.config.sample_rate
        self.n_channels = self.config.n_channels
        
        # 初始化滤波器
        self.filters = self._init_filters()
        
        # ICA 组件（延迟初始化）
        self.ica = None
        self.ica_components = None
        
        logger.info(
            f"EEGProcessor initialized: {self.sample_rate}Hz, "
            f"{self.n_channels} channels"
        )
    
    def _init_filters(self) -> Dict:
        """初始化滤波器组"""
        filters = {}
        
        # 1. 带通滤波器 (0.5-45Hz)
        sos_bp = signal.butter(
            self.config.filter_order,
            [0.5, 45],
            btype='band',
            fs=self.sample_rate,
            output='sos'
        )
        filters['bandpass'] = sos_bp
        
        # 2. 高通滤波器 (0.5Hz) - 去除直流偏移
        sos_hp = signal.butter(
            self.config.filter_order,
            0.5,
            btype='high',
            fs=self.sample_rate,
            output='sos'
        )
        filters['highpass'] = sos_hp
        
        # 3. 低通滤波器 (45Hz) - 去除高频噪声
        sos_lp = signal.butter(
            self.config.filter_order,
            45,
            btype='low',
            fs=self.sample_rate,
            output='sos'
        )
        filters['lowpass'] = sos_lp
        
        # 4. 工频陷波滤波器 (50Hz 或 60Hz)
        sos_notch = signal.iirnotch(
            self.config.notch_freq,
            self.config.notch_q,
            self.sample_rate
        )
        filters['notch'] = sos_notch
        
        logger.info("Filters initialized: bandpass, highpass, lowpass, notch")
        return filters
    
    # ==================== 信号导入 ====================
    
    def load_edf(self, filepath: str) -> Tuple[np.ndarray, List[str]]:
        """
        加载 EDF/EDF+ 文件
        
        Args:
            filepath: EDF 文件路径
            
        Returns:
            (data, channel_names): EEG 数据和通道名称
        """
        try:
            import pyedflib
            
            f = pyedflib.EdfReader(filepath)
            n_channels = f.signals_in_file
            
            # 获取通道标签
            channel_labels = [f.getSignalLabels()[i] for i in range(n_channels)]
            
            # 读取数据
            data = np.zeros((n_channels, f.getNSamples()[0]))
            for i in range(n_channels):
                data[i, :] = f.readSignal(i)
            
            f.close()
            
            logger.info(
                f"Loaded EDF file: {filepath}, "
                f"shape: {data.shape}, channels: {channel_labels}"
            )
            return data, channel_labels
            
        except Exception as e:
            logger.error(f"Failed to load EDF file: {e}")
            raise
    
    def load_csv(self, filepath: str) -> Tuple[np.ndarray, List[str]]:
        """
        加载 CSV 格式的 EEG 数据
        
        Args:
            filepath: CSV 文件路径
            
        Returns:
            (data, channel_names)
        """
        import pandas as pd
        
        df = pd.read_csv(filepath)
        
        # 假设第一列是时间，其余是通道
        channel_names = df.columns[1:].tolist()
        data = df.iloc[:, 1:].values.T
        
        logger.info(f"Loaded CSV file: {filepath}, shape: {data.shape}")
        return data, channel_names
    
    def load_mne(self, filepath: str) -> Tuple[np.ndarray, List[str]]:
        """
        使用 MNE-Python 加载各种格式的 EEG 数据
        
        支持格式：EDF, BDF, GDF, SET (EEGLab), FIF (MNE)
        
        Args:
            filepath: 数据文件路径
            
        Returns:
            (data, channel_names)
        """
        try:
            import mne
            
            # 自动检测格式并加载
            raw = mne.io.read_raw_edf(filepath, preload=True)
            
            # 重采样到目标采样率（如果需要）
            if raw.info['sfreq'] != self.sample_rate:
                raw = raw.resample(self.sample_rate)
            
            data = raw.get_data()
            channel_names = raw.ch_names
            
            logger.info(
                f"Loaded with MNE: {filepath}, "
                f"shape: {data.shape}, sfreq: {raw.info['sfreq']}Hz"
            )
            return data, channel_names
            
        except Exception as e:
            logger.error(f"MNE loading failed: {e}")
            # 回退到 pyedflib
            return self.load_edf(filepath)
    
    # ==================== 信号预处理 ====================
    
    def preprocess(
        self,
        data: np.ndarray,
        remove_artifacts: bool = True,
        method: str = 'threshold'
    ) -> np.ndarray:
        """
        预处理 EEG 信号
        
        步骤:
        1. 带通滤波 (0.5-45Hz)
        2. 工频滤波 (50Hz 陷波)
        3. 伪迹去除
        
        Args:
            data: 原始 EEG 数据 (n_channels, n_samples)
            remove_artifacts: 是否去除伪迹
            method: 伪迹去除方法 ('threshold', 'ica', 'asr')
            
        Returns:
            预处理后的数据
        """
        # 1. 带通滤波
        filtered = signal.sosfiltfilt(
            self.filters['bandpass'],
            data,
            axis=-1
        )
        
        # 2. 工频滤波
        cleaned = signal.sosfiltfilt(
            self.filters['notch'][0],
            filtered,
            axis=-1
        )
        
        # 3. 伪迹去除
        if remove_artifacts:
            cleaned = self.remove_artifacts(cleaned, method=method)
        
        logger.info(
            f"Preprocessing complete: {data.shape} -> {cleaned.shape}, "
            f"method: {method}"
        )
        return cleaned
    
    def remove_artifacts(
        self,
        data: np.ndarray,
        method: str = 'threshold',
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        去除伪迹（眼动、肌电、心电等）
        
        支持方法:
        - threshold: 简单阈值法
        - ica: 独立成分分析
        - asr: 自适应子空间重建
        
        Args:
            data: EEG 数据
            method: 去除方法
            threshold: 阈值（μV），默认使用配置值
            
        Returns:
            去伪迹后的数据
        """
        threshold = threshold or self.config.artifact_threshold
        
        if method == 'threshold':
            return self._remove_artifacts_threshold(data, threshold)
        elif method == 'ica':
            return self._remove_artifacts_ica(data)
        elif method == 'asr':
            return self._remove_artifacts_asr(data)
        else:
            logger.warning(f"Unknown artifact removal method: {method}")
            return data
    
    def _remove_artifacts_threshold(
        self,
        data: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """阈值法去除伪迹"""
        cleaned = data.copy()
        
        for i in range(data.shape[0]):
            mask = np.abs(data[i]) > threshold
            if np.any(mask):
                # 线性插值替换异常值
                cleaned[i, mask] = np.interp(
                    np.where(mask)[0],
                    np.where(~mask)[0],
                    data[i, ~mask]
                )
        
        logger.info(f"Threshold artifact removal: {threshold}μV")
        return cleaned
    
    def _remove_artifacts_ica(self, data: np.ndarray) -> np.ndarray:
        """
        ICA 独立成分分析去除伪迹
        
        自动识别并去除眼电（EOG）、肌电（EMG）成分
        """
        try:
            from sklearn.decomposition import FastICA
            
            # 转置为 (samples, channels)
            data_T = data.T
            
            # ICA 分解
            ica = FastICA(n_components=min(data.shape[0], data.shape[1]))
            components = ica.fit_transform(data_T)
            
            # 识别伪迹成分（基于频谱特征）
            artifact_components = self._identify_ica_artifacts(components)
            
            # 重建信号（去除伪迹成分）
            clean_components = components.copy()
            clean_components[:, artifact_components] = 0
            
            # 逆变换回信号空间
            cleaned_T = ica.inverse_transform(clean_components)
            cleaned = cleaned_T.T
            
            self.ica = ica
            self.ica_components = components
            
            logger.info(
                f"ICA artifact removal: removed {len(artifact_components)} components"
            )
            return cleaned
            
        except Exception as e:
            logger.error(f"ICA artifact removal failed: {e}")
            return self._remove_artifacts_threshold(data, self.config.artifact_threshold)
    
    def _identify_ica_artifacts(self, components: np.ndarray) -> List[int]:
        """
        识别 ICA 成分中的伪迹
        
        基于:
        - 高频能量（肌电）
        - 低频能量（眼电）
        - 幅度异常
        """
        artifact_indices = []
        
        for i in range(components.shape[1]):
            component = components[:, i]
            
            # 计算频谱
            freqs, psd = signal.welch(component, self.sample_rate)
            
            # 眼电：低频能量占比高
            delta_mask = freqs < 4
            theta_mask = (freqs >= 4) & (freqs < 8)
            delta_power = np.mean(psd[delta_mask])
            theta_power = np.mean(psd[theta_mask])
            
            # 肌电：高频能量占比高
            gamma_mask = freqs > 30
            gamma_power = np.mean(psd[gamma_mask])
            
            # 判断是否为伪迹
            if delta_power > np.mean(psd) * 3:  # 强低频
                artifact_indices.append(i)
            elif gamma_power > np.mean(psd) * 2:  # 强高频
                artifact_indices.append(i)
        
        return artifact_indices
    
    def _remove_artifacts_asr(self, data: np.ndarray) -> np.ndarray:
        """
        自适应子空间重建 (ASR)
        
        基于统计方法检测并修复异常段
        """
        # 简化的 ASR 实现
        cleaned = data.copy()
        
        # 计算滑动窗口统计
        window_size = self.sample_rate  # 1 秒窗口
        step = self.sample_rate // 4  # 250ms 步长
        
        for i in range(0, data.shape[1] - window_size, step):
            window = data[:, i:i+window_size]
            
            # 计算窗口标准差
            std = np.std(window)
            
            # 如果标准差异常，使用前后窗口插值
            if std > self.config.artifact_threshold / 2:
                if i > 0 and i + window_size < data.shape[1]:
                    prev_window = data[:, i-window_size:i]
                    next_window = data[:, i+window_size:i+window_size*2]
                    cleaned[:, i:i+window_size] = (prev_window + next_window) / 2
        
        logger.info("ASR artifact removal completed")
        return cleaned
    
    # ==================== 特征提取 ====================
    
    def extract_features(self, data: np.ndarray) -> Dict:
        """
        提取 EEG 特征
        
        Args:
            data: 预处理后的 EEG 数据
            
        Returns:
            特征字典
        """
        features = {
            'power_spectrum': self.calc_power_spectrum(data),
            'band_power': self.calc_band_power(data),
            'coherence': self.calc_coherence(data),
            'asymmetry': self.calc_asymmetry(data),
            'spectral_entropy': self.calc_spectral_entropy(data),
            'microstates': self.calc_microstates(data),
        }
        
        logger.info(f"Features extracted: {list(features.keys())}")
        return features
    
    def calc_power_spectrum(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率谱密度 (PSD)
        
        使用 Welch 方法
        
        Returns:
            (frequencies, psd)
        """
        freqs, psd = signal.welch(
            data,
            self.sample_rate,
            nperseg=self.sample_rate * 2,  # 2 秒窗口
            axis=-1
        )
        return freqs, psd
    
    def calc_band_power(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算各频段功率
        
        Returns:
            各频段功率字典 {delta, theta, alpha, beta, gamma}
        """
        freqs, psd = self.calc_power_spectrum(data)
        
        band_power = {}
        for band, (low, high) in self.config.bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_power[band] = np.mean(psd[:, mask], axis=-1)
        
        return band_power
    
    def calc_coherence(self, data: np.ndarray) -> np.ndarray:
        """
        计算通道间相干性
        
        Returns:
            相干性矩阵 (n_channels, n_channels)
        """
        n_channels = data.shape[0]
        coherence = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                f, coh = signal.coherence(
                    data[i],
                    data[j],
                    self.sample_rate,
                    nperseg=self.sample_rate
                )
                # 取平均相干性
                coherence[i, j] = np.mean(coh)
                coherence[j, i] = coherence[i, j]
        
        return coherence
    
    def calc_asymmetry(self, data: np.ndarray) -> Dict[str, float]:
        """
        计算左右脑不对称性
        
        基于前额叶通道的频段功率差异
        
        Returns:
            各频段不对称性
        """
        band_power = self.calc_band_power(data)
        
        # 假设通道 0-6 为左脑，7-13 为右脑
        left_idx = range(0, min(7, data.shape[0]))
        right_idx = range(min(7, data.shape[0]), data.shape[0])
        
        asymmetry = {}
        for band, power in band_power.items():
            left_power = np.mean(power[left_idx])
            right_power = np.mean(power[right_idx])
            asymmetry[band] = np.log10(right_power + 1e-10) - np.log10(left_power + 1e-10)
        
        return asymmetry
    
    def calc_spectral_entropy(self, data: np.ndarray) -> np.ndarray:
        """
        计算谱熵（信号复杂度指标）
        
        Returns:
            各通道的谱熵
        """
        freqs, psd = self.calc_power_spectrum(data)
        
        # 归一化 PSD
        psd_norm = psd / (np.sum(psd, axis=-1, keepdims=True) + 1e-10)
        
        # 计算熵
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=-1)
        
        return entropy
    
    def calc_microstates(self, data: np.ndarray, n_states: int = 4) -> Dict:
        """
        计算 EEG 微状态
        
        使用 K-means 聚类识别稳定的拓扑模式
        
        Args:
            data: EEG 数据
            n_states: 微状态数量
            
        Returns:
            微状态分析结果
        """
        from sklearn.cluster import KMeans
        
        # 转置为 (samples, channels)
        data_T = data.T
        
        # K-means 聚类
        kmeans = KMeans(n_clusters=n_states, random_state=42)
        labels = kmeans.fit_predict(data_T)
        
        # 计算每个微状态的持续时间
        durations = []
        current_state = labels[0]
        duration = 1
        
        for i in range(1, len(labels)):
            if labels[i] == current_state:
                duration += 1
            else:
                durations.append(duration)
                current_state = labels[i]
                duration = 1
        durations.append(duration)
        
        # 计算覆盖度（每个状态的时间占比）
        coverage = {}
        for i in range(n_states):
            coverage[i] = np.sum(labels == i) / len(labels)
        
        return {
            'centroids': kmeans.cluster_centers_,
            'labels': labels.tolist(),
            'durations': durations,
            'mean_duration': np.mean(durations),
            'coverage': coverage,
        }
    
    # ==================== 时频分析 ====================
    
    def time_frequency_analysis(
        self,
        data: np.ndarray,
        method: str = 'wavelet'
    ) -> Dict:
        """
        时频分析
        
        Args:
            data: EEG 数据
            method: 分析方法 ('wavelet', 'stft')
            
        Returns:
            时频分析结果
        """
        if method == 'wavelet':
            return self._wavelet_transform(data)
        elif method == 'stft':
            return self._stft(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _wavelet_transform(self, data: np.ndarray) -> Dict:
        """
        连续小波变换
        
        Returns:
            各频段的小波系数
        """
        try:
            import pywt
            
            # 定义小波
            wavelet = 'morl'
            
            # 各频段中心频率
            center_freqs = {
                'delta': 2,
                'theta': 6,
                'alpha': 10,
                'beta': 20,
                'gamma': 40,
            }
            
            result = {}
            for band, freq in center_freqs.items():
                # 计算小波尺度
                scale = pywt.central_frequency(wavelet) * self.sample_rate / freq
                
                # 小波变换
                coeffs = pywt.cwt(data, scale, wavelet)[0]
                result[band] = np.abs(coeffs)
            
            return result
            
        except Exception as e:
            logger.error(f"Wavelet transform failed: {e}")
            return {}
    
    def _stft(self, data: np.ndarray) -> Dict:
        """
        短时傅里叶变换 (STFT)
        
        Returns:
            时频谱
        """
        f, t, Zxx = signal.stft(
            data,
            self.sample_rate,
            nperseg=self.sample_rate // 4,
            axis=-1
        )
        
        return {
            'frequencies': f,
            'times': t,
            'spectrogram': np.abs(Zxx),
        }
    
    # ==================== AI 分析 ====================
    
    def analyze(self, data: np.ndarray) -> Dict:
        """
        AI 分析：专注度、情绪等
        
        Args:
            data: EEG 数据
            
        Returns:
            分析结果
        """
        features = self.extract_features(data)
        
        # 专注度计算
        attention_score = self._calc_attention(features)
        
        # 放松度计算
        relaxation_score = self._calc_relaxation(features)
        
        # 情绪识别
        valence, arousal = self._calc_emotion(features)
        
        # 认知负荷
        cognitive_load = self._calc_cognitive_load(features)
        
        return {
            'attention_score': float(attention_score),
            'relaxation_score': float(relaxation_score),
            'valence': float(valence),
            'arousal': float(arousal),
            'cognitive_load': float(cognitive_load),
            'band_power': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in features['band_power'].items()
            },
            'asymmetry': features['asymmetry'],
            'spectral_entropy': features['spectral_entropy'].tolist(),
        }
    
    def _calc_attention(self, features: Dict) -> float:
        """
        计算专注度分数
        
        基于 beta/(alpha+theta) 比率
        """
        band_power = features['band_power']
        
        alpha = np.mean(band_power['alpha'])
        beta = np.mean(band_power['beta'])
        theta = np.mean(band_power['theta'])
        
        # 专注度 = beta / (alpha + theta)
        ratio = beta / (alpha + theta + 1e-6)
        
        # 映射到 0-100
        score = np.clip(ratio * 10, 0, 100)
        
        return score
    
    def _calc_relaxation(self, features: Dict) -> float:
        """
        计算放松度分数
        
        基于 alpha/beta 比率
        """
        band_power = features['band_power']
        
        alpha = np.mean(band_power['alpha'])
        beta = np.mean(band_power['beta'])
        
        ratio = alpha / (beta + 1e-6)
        score = np.clip(ratio * 10, 0, 100)
        
        return score
    
    def _calc_emotion(self, features: Dict) -> Tuple[float, float]:
        """
        计算情绪维度（valence, arousal）
        
        - Valence: 愉悦度（基于左右脑不对称性）
        - Arousal: 唤醒度（基于 beta/alpha 比率）
        """
        asymmetry = features['asymmetry']
        band_power = features['band_power']
        
        # Valence: 前额叶 alpha 不对称性
        valence = np.tanh(asymmetry['alpha'])
        
        # Arousal: beta/alpha 比率
        beta = np.mean(band_power['beta'])
        alpha = np.mean(band_power['alpha'])
        arousal = np.clip(beta / (alpha + 1e-6) / 5, 0, 1)
        
        return valence, arousal
    
    def _calc_cognitive_load(self, features: Dict) -> float:
        """
        计算认知负荷
        
        基于 theta/beta 比率和谱熵
        """
        band_power = features['band_power']
        entropy = features['spectral_entropy']
        
        theta = np.mean(band_power['theta'])
        beta = np.mean(band_power['beta'])
        
        # 认知负荷 = theta/beta + 归一化熵
        ratio = theta / (beta + 1e-6)
        norm_entropy = np.mean(entropy) / np.log(self.sample_rate // 2)
        
        load = (ratio + norm_entropy) / 2
        load = np.clip(load * 100, 0, 100)
        
        return load
    
    # ==================== 数据导出 ====================
    
    def export_features(self, features: Dict, filepath: str, format: str = 'json'):
        """
        导出特征到文件
        
        Args:
            features: 特征字典
            filepath: 输出文件路径
            format: 文件格式 ('json', 'csv', 'npy')
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(features, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(features)
            df.to_csv(filepath, index=False)
        
        elif format == 'npy':
            np.save(filepath, features)
        
        logger.info(f"Features exported to {filepath}")


# ==================== 便捷函数 ====================

def process_eeg_file(
    filepath: str,
    sample_rate: int = 256,
    remove_artifacts: bool = True,
    artifact_method: str = 'ica'
) -> Dict:
    """
    处理 EEG 文件的便捷函数
    
    Args:
        filepath: EDF 文件路径
        sample_rate: 采样率
        remove_artifacts: 是否去除伪迹
        artifact_method: 伪迹去除方法
        
    Returns:
        分析结果
    """
    config = EEGConfig(sample_rate=sample_rate)
    processor = EEGProcessor(config)
    
    # 加载数据
    data, channels = processor.load_edf(filepath)
    
    # 预处理
    cleaned = processor.preprocess(
        data,
        remove_artifacts=remove_artifacts,
        method=artifact_method
    )
    
    # 分析
    result = processor.analyze(cleaned)
    result['channels'] = channels
    result['duration'] = len(data[0]) / sample_rate
    
    return result


def generate_synthetic_eeg(
    duration: int = 60,
    sample_rate: int = 256,
    n_channels: int = 14,
    attention_level: float = 0.5
) -> np.ndarray:
    """
    生成模拟 EEG 数据（用于测试）
    
    Args:
        duration: 时长（秒）
        sample_rate: 采样率
        n_channels: 通道数
        attention_level: 专注度水平 (0-1)
        
    Returns:
        模拟 EEG 数据
    """
    t = np.linspace(0, duration, duration * sample_rate)
    data = np.zeros((n_channels, len(t)))
    
    # 各频段振幅（基于专注度调整）
    alpha_amp = 20 * (1 - attention_level)
    beta_amp = 10 * (1 + attention_level)
    theta_amp = 15 * (1 - attention_level * 0.5)
    
    for i in range(n_channels):
        # 添加各频段振荡
        data[i] = (
            np.sin(2 * np.pi * 10 * t) * alpha_amp +  # α波
            np.sin(2 * np.pi * 20 * t) * beta_amp +   # β波
            np.sin(2 * np.pi * 6 * t) * theta_amp +   # θ波
            np.random.randn(len(t)) * 5  # 噪声
        )
        
        # 添加通道间差异
        data[i] *= (1 + 0.1 * np.random.randn())
    
    return data


if __name__ == "__main__":
    import sys
    
    # 测试模式
    if len(sys.argv) > 1:
        # 处理真实文件
        result = process_eeg_file(sys.argv[1])
        print("\n=== EEG Analysis Result ===")
        print(f"Attention: {result['attention_score']:.1f}/100")
        print(f"Relaxation: {result['relaxation_score']:.1f}/100")
        print(f"Valence: {result['valence']:.2f}")
        print(f"Arousal: {result['arousal']:.2f}")
        print(f"Cognitive Load: {result['cognitive_load']:.1f}/100")
    else:
        # 使用模拟数据测试
        print("Testing with synthetic EEG data...")
        
        config = EEGConfig(sample_rate=256, n_channels=14)
        processor = EEGProcessor(config)
        
        # 生成模拟数据
        data = generate_synthetic_eeg(
            duration=60,
            attention_level=0.7
        )
        
        # 预处理
        cleaned = processor.preprocess(data, method='ica')
        
        # 分析
        result = processor.analyze(cleaned)
        
        print("\n=== EEG Analysis Result ===")
        print(f"Attention: {result['attention_score']:.1f}/100")
        print(f"Relaxation: {result['relaxation_score']:.1f}/100")
        print(f"Valence: {result['valence']:.2f}")
        print(f"Arousal: {result['arousal']:.2f}")
        print(f"Cognitive Load: {result['cognitive_load']:.1f}/100")
        
        # 特征提取
        features = processor.extract_features(cleaned)
        print(f"\nBand Power: {features['band_power']}")
        print(f"Asymmetry: {features['asymmetry']}")
