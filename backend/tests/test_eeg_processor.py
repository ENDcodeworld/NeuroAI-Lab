"""
Unit Tests for EEG Processor
"""

import pytest
import numpy as np
from app.eeg_processor_v2 import (
    EEGProcessor,
    EEGConfig,
    process_eeg_file,
    generate_synthetic_eeg,
)


class TestEEGConfig:
    """测试配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = EEGConfig()
        assert config.sample_rate == 256
        assert config.n_channels == 14
        assert config.filter_order == 4
        assert config.notch_freq == 50.0
        assert 'alpha' in config.bands
        assert config.bands['alpha'] == (8, 13)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = EEGConfig(
            sample_rate=512,
            n_channels=32,
            notch_freq=60.0
        )
        assert config.sample_rate == 512
        assert config.n_channels == 32
        assert config.notch_freq == 60.0


class TestEEGProcessor:
    """测试 EEG 处理器"""
    
    @pytest.fixture
    def processor(self):
        """创建测试处理器"""
        config = EEGConfig(sample_rate=256, n_channels=14)
        return EEGProcessor(config)
    
    @pytest.fixture
    def synthetic_data(self):
        """生成模拟数据"""
        return generate_synthetic_eeg(
            duration=10,
            sample_rate=256,
            n_channels=14,
            attention_level=0.5
        )
    
    def test_initialization(self, processor):
        """测试初始化"""
        assert processor.sample_rate == 256
        assert processor.n_channels == 14
        assert 'bandpass' in processor.filters
        assert 'notch' in processor.filters
    
    def test_filters_initialized(self, processor):
        """测试滤波器初始化"""
        assert 'bandpass' in processor.filters
        assert 'highpass' in processor.filters
        assert 'lowpass' in processor.filters
        assert 'notch' in processor.filters
    
    def test_preprocess(self, processor, synthetic_data):
        """测试预处理"""
        cleaned = processor.preprocess(synthetic_data)
        
        assert cleaned.shape == synthetic_data.shape
        assert not np.isnan(cleaned).any()
        assert not np.isinf(cleaned).any()
    
    def test_preprocess_with_ica(self, processor, synthetic_data):
        """测试 ICA 伪迹去除"""
        cleaned = processor.preprocess(
            synthetic_data,
            remove_artifacts=True,
            method='ica'
        )
        
        assert cleaned.shape == synthetic_data.shape
    
    def test_preprocess_with_threshold(self, processor, synthetic_data):
        """测试阈值法伪迹去除"""
        cleaned = processor.preprocess(
            synthetic_data,
            remove_artifacts=True,
            method='threshold'
        )
        
        assert cleaned.shape == synthetic_data.shape
    
    def test_extract_features(self, processor, synthetic_data):
        """测试特征提取"""
        cleaned = processor.preprocess(synthetic_data)
        features = processor.extract_features(cleaned)
        
        assert 'power_spectrum' in features
        assert 'band_power' in features
        assert 'coherence' in features
        assert 'asymmetry' in features
        assert 'spectral_entropy' in features
        assert 'microstates' in features
        
        # 检查频段功率
        band_power = features['band_power']
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            assert band in band_power
            assert len(band_power[band]) == 14  # n_channels
    
    def test_calc_band_power(self, processor, synthetic_data):
        """测试频段功率计算"""
        cleaned = processor.preprocess(synthetic_data)
        band_power = processor.calc_band_power(cleaned)
        
        assert len(band_power) == 5
        assert all(len(v) == 14 for v in band_power.values())
    
    def test_calc_coherence(self, processor, synthetic_data):
        """测试相干性计算"""
        cleaned = processor.preprocess(synthetic_data)
        coherence = processor.calc_coherence(cleaned)
        
        assert coherence.shape == (14, 14)
        assert np.allclose(coherence, coherence.T)  # 对称矩阵
        assert np.all((coherence >= 0) & (coherence <= 1))
    
    def test_calc_asymmetry(self, processor, synthetic_data):
        """测试不对称性计算"""
        cleaned = processor.preprocess(synthetic_data)
        asymmetry = processor.calc_asymmetry(cleaned)
        
        assert len(asymmetry) == 5
        for band in asymmetry:
            assert isinstance(asymmetry[band], float)
    
    def test_analyze(self, processor, synthetic_data):
        """测试 AI 分析"""
        cleaned = processor.preprocess(synthetic_data)
        result = processor.analyze(cleaned)
        
        assert 'attention_score' in result
        assert 'relaxation_score' in result
        assert 'valence' in result
        assert 'arousal' in result
        assert 'cognitive_load' in result
        
        # 检查分数范围
        assert 0 <= result['attention_score'] <= 100
        assert 0 <= result['relaxation_score'] <= 100
        assert -1 <= result['valence'] <= 1
        assert 0 <= result['arousal'] <= 1
        assert 0 <= result['cognitive_load'] <= 100
    
    def test_attention_level_effect(self, processor):
        """测试专注度水平对分析结果的影响"""
        # 高专注度数据
        high_attention_data = generate_synthetic_eeg(
            duration=10,
            attention_level=0.8
        )
        high_cleaned = processor.preprocess(high_attention_data)
        high_result = processor.analyze(high_cleaned)
        
        # 低专注度数据
        low_attention_data = generate_synthetic_eeg(
            duration=10,
            attention_level=0.2
        )
        low_cleaned = processor.preprocess(low_attention_data)
        low_result = processor.analyze(low_cleaned)
        
        # 高专注度应该有更高的 attention score
        assert high_result['attention_score'] > low_result['attention_score']
    
    def test_time_frequency_analysis_wavelet(self, processor, synthetic_data):
        """测试小波时频分析"""
        cleaned = processor.preprocess(synthetic_data)
        tf_result = processor.time_frequency_analysis(cleaned, method='wavelet')
        
        # 应该包含各频段的小波系数
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if band in tf_result:
                assert tf_result[band].shape[0] == 14  # n_channels
    
    def test_time_frequency_analysis_stft(self, processor, synthetic_data):
        """测试 STFT 时频分析"""
        cleaned = processor.preprocess(synthetic_data)
        tf_result = processor.time_frequency_analysis(cleaned, method='stft')
        
        assert 'frequencies' in tf_result
        assert 'times' in tf_result
        assert 'spectrogram' in tf_result
    
    def test_microstates(self, processor, synthetic_data):
        """测试微状态分析"""
        cleaned = processor.preprocess(synthetic_data)
        microstates = processor.calc_microstates(cleaned, n_states=4)
        
        assert 'centroids' in microstates
        assert 'labels' in microstates
        assert 'durations' in microstates
        assert 'coverage' in microstates
        
        assert microstates['centroids'].shape == (4, 14)  # 4 states, 14 channels
        assert len(microstates['coverage']) == 4


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_generate_synthetic_eeg(self):
        """测试模拟数据生成"""
        data = generate_synthetic_eeg(
            duration=5,
            sample_rate=256,
            n_channels=14,
            attention_level=0.5
        )
        
        assert data.shape == (14, 5 * 256)
        assert not np.isnan(data).any()
    
    def test_generate_synthetic_eeg_attention_effect(self):
        """测试专注度对模拟数据的影响"""
        high_data = generate_synthetic_eeg(attention_level=0.9)
        low_data = generate_synthetic_eeg(attention_level=0.1)
        
        # 高专注度数据应该有更高的 beta 波成分
        processor = EEGProcessor()
        high_cleaned = processor.preprocess(high_data)
        low_cleaned = processor.preprocess(low_data)
        
        high_bp = processor.calc_band_power(high_cleaned)
        low_bp = processor.calc_band_power(low_cleaned)
        
        # beta 波在高专注度时应该更强
        assert np.mean(high_bp['beta']) > np.mean(low_bp['beta'])


class TestArtifactRemoval:
    """测试伪迹去除"""
    
    @pytest.fixture
    def data_with_artifacts(self):
        """生成带伪迹的数据"""
        data = generate_synthetic_eeg(duration=10)
        
        # 添加眼电伪迹（低频大幅值）
        t = np.arange(data.shape[1]) / 256
        eog = np.sin(2 * np.pi * 2 * t) * 150  # 2Hz, 150μV
        data[0, :] += eog
        data[1, :] += eog
        
        # 添加肌电伪迹（高频）
        emg = np.random.randn(data.shape[1]) * 30
        data[12, :] += emg
        data[13, :] += emg
        
        return data
    
    def test_threshold_artifact_removal(self, processor, data_with_artifacts):
        """测试阈值法伪迹去除"""
        cleaned = processor.remove_artifacts(
            data_with_artifacts,
            method='threshold',
            threshold=100.0
        )
        
        # 伪迹去除后，最大值应该降低
        assert np.max(np.abs(cleaned)) < np.max(np.abs(data_with_artifacts))
    
    def test_ica_artifact_removal(self, processor, data_with_artifacts):
        """测试 ICA 伪迹去除"""
        cleaned = processor.remove_artifacts(
            data_with_artifacts,
            method='ica'
        )
        
        assert cleaned.shape == data_with_artifacts.shape
        # ICA 应该能降低伪迹幅度
        assert np.std(cleaned) < np.std(data_with_artifacts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
