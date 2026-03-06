"""
Attention Analysis Model
专注度分析深度学习模型

模型架构:
- CNN 特征提取
- LSTM 时序建模
- Attention 机制
- 多任务学习（专注度 + 认知负荷）

作者：NeuroAI-Lab Team
版本：1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EEGAttentionModel(nn.Module):
    """
    基于 EEG 的专注度评估模型
    
    架构:
    Input (N, T, C) 
      ↓
    Temporal Conv (N, T, 64)
      ↓
    Spatial Conv (N, T, 128)
      ↓
    BiLSTM (N, T, 256)
      ↓
    Multi-Head Attention (N, T, 256)
      ↓
    Global Pooling (N, 256)
      ↓
    FC Layers
      ↓
    Output: attention_score (0-100), cognitive_load (0-100)
    """
    
    def __init__(
        self,
        n_channels: int = 14,
        seq_length: int = 256,
        hidden_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.3
    ):
        """
        初始化模型
        
        Args:
            n_channels: EEG 通道数
            seq_length: 序列长度（采样点数）
            hidden_dim: 隐藏层维度
            n_heads: Attention 头数
            dropout: Dropout 比率
        """
        super(EEGAttentionModel, self).__init__()
        
        self.n_channels = n_channels
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # 1. 时间卷积层
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 2. 空间卷积层
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 3. BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 4. Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 5. Layer Norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 6. Feed-forward layers
        self.fc_attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.fc_cognitive_load = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, time, channels)
            
        Returns:
            输出字典 {attention_score, cognitive_load, features}
        """
        # x: (batch, time, channels)
        batch_size = x.size(0)
        
        # 1. 时间卷积
        x = x.permute(0, 2, 1)  # (batch, channels, time)
        x = self.temporal_conv(x)  # (batch, 64, time)
        
        # 2. 空间卷积
        x = self.spatial_conv(x)  # (batch, 128, time)
        
        # 3. 转置回 (batch, time, features)
        x = x.permute(0, 2, 1)  # (batch, time, 128)
        
        # 4. BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden_dim)
        
        # 5. Self-Attention
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # 6. Residual + Norm
        x = self.norm1(attn_out + lstm_out)
        
        # 7. Global Average Pooling
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # 8. 输出层
        attention_score = self.fc_attention(x)  # (batch, 1)
        cognitive_load = self.fc_cognitive_load(x)  # (batch, 1)
        
        return {
            'attention_score': attention_score,
            'cognitive_load': cognitive_load,
            'features': x
        }


class EmotionRecognitionModel(nn.Module):
    """
    情绪识别模型（Valence + Arousal）
    
    基于 EEG 频段特征和深度神经网络
    """
    
    def __init__(
        self,
        n_channels: int = 14,
        n_bands: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        初始化情绪识别模型
        
        Args:
            n_channels: EEG 通道数
            n_bands: 频段数量 (delta, theta, alpha, beta, gamma)
            hidden_dim: 隐藏层维度
            dropout: Dropout 比率
        """
        super(EmotionRecognitionModel, self).__init__()
        
        # 输入特征维度：channels × bands
        input_dim = n_channels * n_bands
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Valence 预测头
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Arousal 预测头
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 频段功率特征 (batch, channels, bands)
            
        Returns:
            输出字典 {valence, arousal}
        """
        # 展平特征
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch, channels*bands)
        
        # 共享层
        features = self.shared_layers(x)
        
        # 多任务输出
        valence = self.valence_head(features)
        arousal = self.arousal_head(features)
        
        return {
            'valence': valence,
            'arousal': arousal,
            'features': features
        }


class CombinedBCIModel(nn.Module):
    """
    联合 BCI 模型
    
    同时预测:
    - 专注度
    - 认知负荷
    - 情绪 (valence, arousal)
    """
    
    def __init__(
        self,
        n_channels: int = 14,
        seq_length: int = 256,
        hidden_dim: int = 256
    ):
        """
        初始化联合模型
        
        Args:
            n_channels: EEG 通道数
            seq_length: 序列长度
            hidden_dim: 隐藏层维度
        """
        super(CombinedBCIModel, self).__init__()
        
        # 专注度模型
        self.attention_model = EEGAttentionModel(
            n_channels=n_channels,
            seq_length=seq_length,
            hidden_dim=hidden_dim
        )
        
        # 情绪模型（使用频段特征）
        self.emotion_model = EmotionRecognitionModel(
            n_channels=n_channels,
            n_bands=5
        )
        
        # 频段功率提取（简化版）
        self.band_extractor = BandPowerExtractor(n_channels)
    
    def forward(
        self,
        eeg_data: torch.Tensor,
        use_bands: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            eeg_data: EEG 数据 (batch, time, channels)
            use_bands: 是否使用频段特征
            
        Returns:
            输出字典
        """
        # 专注度分析
        attention_output = self.attention_model(eeg_data)
        
        result = {
            'attention_score': attention_output['attention_score'],
            'cognitive_load': attention_output['cognitive_load'],
        }
        
        # 情绪分析（可选）
        if use_bands:
            band_power = self.band_extractor(eeg_data)
            emotion_output = self.emotion_model(band_power)
            result.update({
                'valence': emotion_output['valence'],
                'arousal': emotion_output['arousal'],
            })
        
        return result


class BandPowerExtractor(nn.Module):
    """
    频段功率特征提取器（可微分版本）
    
    使用 FFT 计算各频段功率
    """
    
    def __init__(
        self,
        n_channels: int = 14,
        sample_rate: int = 256
    ):
        super(BandPowerExtractor, self).__init__()
        
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        
        # 定义频段范围（归一化频率）
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
        }
        
        # 创建频段掩码
        self.register_buffer('band_masks', self._create_band_masks())
    
    def _create_band_masks(self) -> torch.Tensor:
        """创建频段频率掩码"""
        # 简化实现，实际应在 forward 中动态创建
        return torch.zeros(5, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取频段功率
        
        Args:
            x: EEG 数据 (batch, time, channels)
            
        Returns:
            频段功率 (batch, channels, 5)
        """
        # 使用 PyTorch FFT
        x = x.permute(0, 2, 1)  # (batch, channels, time)
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        power = torch.abs(x_fft) ** 2  # 功率谱
        
        # 简化：使用平均 pooling 近似频段功率
        # 实际应使用频率掩码
        n_freqs = power.size(-1)
        
        # 5 个频段的近似索引
        band_indices = [
            (0, n_freqs // 16),      # delta
            (n_freqs // 16, n_freqs // 8),  # theta
            (n_freqs // 8, n_freqs // 4),   # alpha
            (n_freqs // 4, n_freqs // 2),   # beta
            (n_freqs // 2, n_freqs),        # gamma
        ]
        
        band_powers = []
        for start, end in band_indices:
            band_power = power[..., start:end].mean(dim=-1)
            band_powers.append(band_power)
        
        # (batch, channels, 5)
        band_power = torch.stack(band_powers, dim=-1)
        
        return band_power


# ==================== 训练工具函数 ====================

def create_dataloader(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    seq_length: int = 256,
    shuffle: bool = True
):
    """
    创建 PyTorch DataLoader
    
    Args:
        eeg_data: EEG 数据 (samples, channels, time)
        labels: 标签 (samples,)
        batch_size: 批次大小
        seq_length: 序列长度
        shuffle: 是否打乱
        
    Returns:
        DataLoader
    """
    from torch.utils.data import Dataset, DataLoader
    
    class EEGDataset(Dataset):
        def __init__(self, data, labels, seq_length):
            self.data = torch.FloatTensor(data)
            self.labels = torch.FloatTensor(labels)
            self.seq_length = seq_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # 截取或填充到固定长度
            sample = self.data[idx]
            if sample.size(-1) > self.seq_length:
                sample = sample[:, :self.seq_length]
            else:
                pad = torch.zeros(sample.size(0), self.seq_length)
                pad[:, :sample.size(-1)] = sample
                sample = pad
            
            # 转置为 (time, channels)
            sample = sample.permute(1, 0)
            
            return sample, self.labels[idx]
    
    dataset = EEGDataset(eeg_data, labels, seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    return dataloader


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    save_path: Optional[str] = None
):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        save_path: 模型保存路径
        
    Returns:
        训练历史
    """
    model = model.to(device)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # 计算损失
            loss = criterion(
                outputs['attention_score'].squeeze(),
                batch_labels
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                
                loss = criterion(
                    outputs['attention_score'].squeeze(),
                    batch_labels
                )
                
                val_loss += loss.item()
                val_mae += torch.mean(
                    torch.abs(
                        outputs['attention_score'].squeeze() - batch_labels
                    )
                ).item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val MAE: {val_mae:.4f}"
            )
    
    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    return history


# ==================== 模型导出 ====================

def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    output_path: str,
    device: str = 'cpu'
):
    """
    导出模型为 ONNX 格式
    
    Args:
        model: PyTorch 模型
        input_shape: 输入形状 (batch, time, channels)
        output_path: 输出路径
        device: 设备
    """
    model = model.to(device)
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape).to(device)
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['eeg_input'],
        output_names=['attention_score', 'cognitive_load'],
        dynamic_axes={
            'eeg_input': {0: 'batch_size'},
            'attention_score': {0: 'batch_size'},
            'cognitive_load': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to {output_path}")


if __name__ == "__main__":
    # 测试模型
    print("=== Testing EEG Attention Model ===\n")
    
    # 创建模型
    model = EEGAttentionModel(
        n_channels=14,
        seq_length=256,
        hidden_dim=256
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 256, 14)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Attention score shape: {output['attention_score'].shape}")
    print(f"Cognitive load shape: {output['cognitive_load'].shape}")
    print(f"Features shape: {output['features'].shape}")
    
    print("\n✓ Model test passed!")
