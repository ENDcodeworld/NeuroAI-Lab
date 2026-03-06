# EEG 数据集集成指南

本文档介绍如何集成和使用公开的 EEG 数据集。

## 📊 支持的数据集

### 1. PhysioNet EEG Motor Movement/Imagery Dataset

**数据集信息:**
- **被试数量:** 109 人
- **通道数:** 64 通道
- **采样率:** 160 Hz
- **任务类型:** 运动执行 / 运动想象
- **时长:** 每段约 60 秒

**实验范式:**
- 左手握拳
- 右手握拳
- 双脚运动
- 舌头运动

**下载链接:**
https://physionet.org/content/eegmmidb/1.0.0/

**使用方法:**
```python
from app.data_loaders import PhysioNetLoader

# 初始化加载器
loader = PhysioNetLoader('/path/to/physionet')

# 获取被试列表
subjects = loader.get_subjects()

# 加载数据
eeg_data, metadata = loader.load_subject(subject_id=1)

print(f"数据形状：{eeg_data.shape}")
print(f"通道：{metadata['channel_names']}")
```

---

### 2. DEAP Dataset (情绪)

**数据集信息:**
- **被试数量:** 32 人
- **通道数:** 32 通道
- **采样率:** 512 Hz (原始) / 128 Hz (降采样)
- **Trial 数量:** 40 个/人
- **Trial 时长:** 63 秒

**标注维度:**
- Valence (愉悦度): 1-9
- Arousal (唤醒度): 1-9
- Dominance (支配度): 1-9
- Liking (喜好度): 1-9

**下载链接:**
http://www.eecs.qmul.ac.uk/mmv/datasets/deap/

**使用方法:**
```python
from app.data_loaders import DEAPLoader

loader = DEAPLoader('/path/to/deap')

# 加载被试数据
eeg_data, metadata = loader.load_subject(subject_id=1)

# 数据形状：(trials, channels, samples)
print(f"Trial 数量：{eeg_data.shape[0]}")
print(f"标注：{metadata['labels'].keys()}")
```

---

### 3. SEED Dataset (情绪)

**数据集信息:**
- **被试数量:** 15 人
- **通道数:** 62 通道
- **采样率:** 1000 Hz
- **Sessions:** 3 次（不同日期）
- **情绪类别:** 3 类

**情绪类别:**
1. Negative (负面)
2. Neutral (中性)
3. Positive (正面)

**下载链接:**
https://bcmi.sjtu.edu.cn/~seed/

**使用方法:**
```python
from app.data_loaders import SEEDLoader

loader = SEEDLoader('/path/to/seed')

# 加载特定 session 的数据
eeg_data, metadata = loader.load_subject(
    subject_id=1,
    session=1
)

print(f"情绪类别：{metadata['emotion_classes']}")
print(f"标注分布：{np.bincount(metadata['labels'])}")
```

---

## 🔧 数据集管理器

使用 `DatasetManager` 统一管理多个数据集：

```python
from app.data_loaders import create_dataset_manager

# 创建管理器
manager = create_dataset_manager(
    physionet_dir='/path/to/physionet',
    deap_dir='/path/to/deap',
    seed_dir='/path/to/seed'
)

# 列出所有数据集
datasets = manager.list_datasets()
print(f"可用数据集：{datasets}")

# 获取数据集信息
info = manager.get_dataset_info('deap')
print(info)

# 加载数据
eeg_data, metadata = manager.load_data(
    dataset_name='deap',
    subject_id=5
)
```

---

## 📝 数据预处理流程

### 标准预处理管道

```python
from app.eeg_processor_v2 import EEGProcessor, EEGConfig
from app.data_loaders import DEAPLoader

# 1. 加载数据
loader = DEAPLoader('/path/to/deap')
eeg_data, metadata = loader.load_subject(1)

# 2. 配置处理器
config = EEGConfig(
    sample_rate=metadata['sample_rate'],
    n_channels=metadata['n_channels']
)
processor = EEGProcessor(config)

# 3. 预处理
# 选择一个 trial
trial_data = eeg_data[0]  # (channels, samples)

# 带通滤波 + 伪迹去除
cleaned = processor.preprocess(
    trial_data,
    remove_artifacts=True,
    method='ica'  # 或 'threshold', 'asr'
)

# 4. 特征提取
features = processor.extract_features(cleaned)

print(f"频段功率：{features['band_power']}")
print(f"相干性：{features['coherence'].shape}")
```

### 批量预处理

```python
def preprocess_dataset(loader, output_dir):
    """批量预处理整个数据集"""
    from pathlib import Path
    import numpy as np
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subjects = loader.get_subjects()
    
    for subject_id in subjects:
        print(f"Processing subject {subject_id}...")
        
        # 加载数据
        eeg_data, metadata = loader.load_subject(subject_id)
        
        # 预处理
        config = EEGConfig(sample_rate=metadata['sample_rate'])
        processor = EEGProcessor(config)
        
        if len(eeg_data.shape) == 3:
            # DEAP: (trials, channels, samples)
            processed_trials = []
            for trial_idx in range(eeg_data.shape[0]):
                trial = eeg_data[trial_idx]
                cleaned = processor.preprocess(trial)
                processed_trials.append(cleaned)
            
            processed_data = np.array(processed_trials)
        else:
            # PhysioNet: (channels, samples)
            processed_data = processor.preprocess(eeg_data)
        
        # 保存
        np.save(
            output_path / f"subject_{subject_id:03d}.npy",
            processed_data
        )
        
        # 保存元数据
        with open(
            output_path / f"subject_{subject_id:03d}_meta.json",
            'w'
        ) as f:
            json.dump(metadata, f, indent=2, default=str)
    
    print(f"Preprocessing complete. Saved to {output_path}")
```

---

## 📊 特征提取示例

### 频段功率分析

```python
# 计算各频段功率
band_power = processor.calc_band_power(cleaned)

for band, power in band_power.items():
    print(f"{band}: {np.mean(power):.2f} μV²")
```

### 脑区不对称性

```python
# 计算左右脑不对称性
asymmetry = processor.calc_asymmetry(cleaned)

print("左右脑不对称性:")
for band, value in asymmetry.items():
    print(f"  {band}: {value:.3f}")
```

### 功能连接（相干性）

```python
# 计算通道间相干性
coherence = processor.calc_coherence(cleaned)

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(coherence, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(label='Coherence')
plt.title('EEG Functional Connectivity')
plt.xlabel('Channel')
plt.ylabel('Channel')
plt.show()
```

---

## 🎯 情绪识别示例

使用 DEAP 数据集训练情绪分类器：

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. 加载所有被试数据
all_features = []
all_labels = []

for subject_id in range(1, 33):
    eeg_data, metadata = loader.load_subject(subject_id)
    
    for trial_idx in range(eeg_data.shape[0]):
        # 预处理
        trial = eeg_data[trial_idx]
        cleaned = processor.preprocess(trial)
        
        # 提取特征
        features = processor.extract_features(cleaned)
        
        # 使用频段功率作为特征
        feature_vector = np.concatenate([
            features['band_power'][band]
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']
        ])
        
        all_features.append(feature_vector)
        
        # 获取情绪标注（二分类：高/低 valence）
        valence = metadata['labels']['labels'][trial_idx, 0]
        label = 1 if valence >= 5 else 0
        all_labels.append(label)

# 2. 准备训练数据
X = np.array(all_features)
y = np.array(all_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 训练分类器
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X_train, y_train)

# 4. 评估
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📦 数据导出

### 导出为 MNE-Python 格式

```python
import mne

# 创建 Raw 对象
info = mne.create_info(
    ch_names=metadata['channel_names'],
    sfreq=metadata['sample_rate'],
    ch_types='eeg'
)

raw = mne.io.RawArray(cleaned, info)

# 保存为 FIF 格式
raw.save('subject_001_raw.fif')
```

### 导出为 CSV

```python
import pandas as pd

# 创建 DataFrame
df = pd.DataFrame(
    cleaned.T,
    columns=metadata['channel_names']
)

# 添加时间戳
df['time'] = np.arange(len(df)) / metadata['sample_rate']

# 保存
df.to_csv('subject_001_eeg.csv', index=False)
```

---

## 🔍 数据质量控制

### 信噪比评估

```python
def calculate_snr(data, sample_rate):
    """计算信噪比"""
    from scipy import signal
    
    # 信号功率 (0.5-45Hz)
    sos_bp = signal.butter(4, [0.5, 45], btype='band', 
                           fs=sample_rate, output='sos')
    signal_data = signal.sosfiltfilt(sos_bp, data, axis=-1)
    signal_power = np.mean(signal_data ** 2)
    
    # 噪声功率 (>45Hz)
    sos_hp = signal.butter(4, 45, btype='high', 
                           fs=sample_rate, output='sos')
    noise_data = signal.sosfiltfilt(sos_hp, data, axis=-1)
    noise_power = np.mean(noise_data ** 2)
    
    # SNR (dB)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return snr

# 评估数据质量
snr = calculate_snr(cleaned, metadata['sample_rate'])
print(f"信噪比：{snr:.2f} dB")
```

---

## 📚 参考文献

1. **PhysioNet:**
   - Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. 2000.
   - https://physionet.org/

2. **DEAP:**
   - Koelstra S, et al. DEAP: A Database for Emotion Analysis Using Physiological Signals. IEEE TAC. 2012.
   - http://www.eecs.qmul.ac.uk/mmv/datasets/deap/

3. **SEED:**
   - Zheng WL, Lu BL. Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition. IEEE TAC. 2015.
   - https://bcmi.sjtu.edu.cn/~seed/

---

**最后更新:** 2026-03-06  
**作者:** NeuroAI-Lab Team
