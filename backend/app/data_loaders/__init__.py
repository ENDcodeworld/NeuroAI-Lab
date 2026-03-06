"""
EEG Dataset Loaders
公开 EEG 数据集加载器

支持的数据集:
1. PhysioNet EEG Motor Movement/Imagery Dataset
2. DEAP Dataset (情绪)
3. SEED Dataset (情绪)
4. BCI Competition IV Datasets
"""

import os
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


class BaseDatasetLoader(ABC):
    """数据集加载器基类"""
    
    def __init__(self, data_dir: str):
        """
        初始化加载器
        
        Args:
            data_dir: 数据集目录
        """
        self.data_dir = Path(data_dir)
        self.metadata = {}
    
    @abstractmethod
    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, Dict]:
        """
        加载单个被试的数据
        
        Args:
            subject_id: 被试 ID
            
        Returns:
            (eeg_data, metadata)
        """
        pass
    
    @abstractmethod
    def get_subjects(self) -> List[int]:
        """获取所有被试 ID 列表"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """获取数据集信息"""
        pass


class PhysioNetLoader(BaseDatasetLoader):
    """
    PhysioNet EEG Motor Movement/Imagery Dataset 加载器
    
    数据集信息:
    - 109 名被试
    - 64 通道 EEG
    - 160Hz 采样率
    - 任务：运动想象（左手、右手、双脚、舌头）
    
    下载链接:
    https://physionet.org/content/eegmmidb/1.0.0/
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.sample_rate = 160
        self.n_channels = 64
        self.channel_names = self._get_channel_names()
    
    def _get_channel_names(self) -> List[str]:
        """获取通道名称"""
        # 标准 10-20 系统 64 通道
        return [
            'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
            'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
            'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
            'O1', 'OZ', 'O2',
            'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
            'F9', 'F10', 'A1', 'A2'
        ][:64]
    
    def get_subjects(self) -> List[int]:
        """获取被试 ID 列表 (S001-S109)"""
        subjects = []
        for i in range(1, 110):
            subject_dir = self.data_dir / f"S{i:03d}"
            if subject_dir.exists():
                subjects.append(i)
        return subjects
    
    def load_subject(
        self,
        subject_id: int,
        run_type: str = 'motor'
    ) -> Tuple[np.ndarray, Dict]:
        """
        加载被试数据
        
        Args:
            subject_id: 被试 ID (1-109)
            run_type: 任务类型 ('motor', 'imagery')
            
        Returns:
            (eeg_data, metadata)
        """
        subject_dir = self.data_dir / f"S{subject_id:03d}"
        
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject {subject_id} not found")
        
        # 查找对应的 EDF 文件
        edf_files = list(subject_dir.glob("*.edf"))
        
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found for subject {subject_id}")
        
        # 加载所有 runs
        all_data = []
        all_events = []
        
        for edf_file in edf_files:
            try:
                data, events = self._load_edf_file(edf_file)
                all_data.append(data)
                all_events.extend(events)
            except Exception as e:
                logger.warning(f"Failed to load {edf_file}: {e}")
        
        if not all_data:
            raise ValueError(f"No data loaded for subject {subject_id}")
        
        # 合并数据
        eeg_data = np.concatenate(all_data, axis=-1)
        
        metadata = {
            'subject_id': subject_id,
            'dataset': 'PhysioNet EEG Motor Movement/Imagery',
            'sample_rate': self.sample_rate,
            'n_channels': self.n_channels,
            'channel_names': self.channel_names,
            'duration': len(eeg_data[0]) / self.sample_rate,
            'n_runs': len(all_data),
            'events': all_events,
        }
        
        logger.info(
            f"Loaded PhysioNet subject {subject_id}: "
            f"{eeg_data.shape}, {len(all_events)} events"
        )
        
        return eeg_data, metadata
    
    def _load_edf_file(self, filepath: Path) -> Tuple[np.ndarray, List]:
        """加载单个 EDF 文件"""
        import pyedflib
        
        f = pyedflib.EdfReader(str(filepath))
        n_channels = f.signals_in_file
        
        # 读取数据
        data = np.zeros((n_channels, f.getNSamples()[0]))
        for i in range(n_channels):
            data[i, :] = f.readSignal(i)
        
        # 读取事件标注
        events = []
        for i in range(f.annotations_in_file):
            annot = f.readAnnotation(i)
            events.append({
                'onset': annot[0],
                'duration': annot[1],
                'description': annot[2]
            })
        
        f.close()
        
        return data, events
    
    def get_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'name': 'PhysioNet EEG Motor Movement/Imagery',
            'n_subjects': 109,
            'n_channels': 64,
            'sample_rate': 160,
            'tasks': ['motor', 'imagery'],
            'url': 'https://physionet.org/content/eegmmidb/1.0.0/',
            'description': (
                'EEG data from 109 volunteers performing motor/imagery tasks. '
                'Includes left/right fist clenching and feet/tongue movement.'
            )
        }


class DEAPLoader(BaseDatasetLoader):
    """
    DEAP Dataset 加载器
    
    数据集信息:
    - 32 名被试
    - 32 通道 EEG
    - 512Hz 采样率（原始），128Hz（降采样后）
    - 40 个 trials（音乐视频）
    - 标注：valence, arousal, dominance, liking
    
    下载链接:
    http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.sample_rate = 128  # 降采样后
        self.n_channels = 32
        self.n_trials = 40
        
        # DEAP 通道名称
        self.channel_names = [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8',
            'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fz', 'Cz'
        ]
    
    def get_subjects(self) -> List[int]:
        """获取被试 ID 列表 (1-32)"""
        subjects = []
        for i in range(1, 33):
            subject_file = self.data_dir / f"{i:02d}.dat"
            if subject_file.exists():
                subjects.append(i)
        return subjects
    
    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, Dict]:
        """
        加载被试数据
        
        Args:
            subject_id: 被试 ID (1-32)
            
        Returns:
            (eeg_data, metadata)
        """
        # DEAP 数据文件
        data_file = self.data_dir / f"{subject_id:02d}.dat"
        label_file = self.data_dir / f"{subject_id:02d}.labels"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Subject {subject_id} data not found")
        
        try:
            import pickle
            
            # 加载数据
            with open(data_file, 'rb') as f:
                # DEAP 使用 pickle 格式
                data = pickle.load(f, encoding='latin1')
            
            # 提取 EEG 数据 (trials, channels, samples)
            eeg_data = data['data']  # Shape: (40, 32, 8064)
            
            # 重采样到 128Hz（如果原始是 512Hz）
            if eeg_data.shape[-1] == 8064:  # 512Hz * 63s
                from scipy.signal import resample
                eeg_data = resample(eeg_data, 128 * 63, axis=-1)
            
            # 加载标注
            labels = None
            if label_file.exists():
                with open(label_file, 'rb') as f:
                    labels = pickle.load(f, encoding='latin1')
            
            metadata = {
                'subject_id': subject_id,
                'dataset': 'DEAP',
                'sample_rate': self.sample_rate,
                'n_channels': self.n_channels,
                'n_trials': self.n_trials,
                'channel_names': self.channel_names,
                'trial_duration': 63,  # seconds
                'labels': labels,
            }
            
            logger.info(
                f"Loaded DEAP subject {subject_id}: "
                f"shape: {eeg_data.shape}"
            )
            
            return eeg_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load DEAP subject {subject_id}: {e}")
            raise
    
    def get_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'name': 'DEAP',
            'n_subjects': 32,
            'n_channels': 32,
            'sample_rate': 128,
            'n_trials': 40,
            'trial_duration': 63,
            'url': 'http://www.eecs.qmul.ac.uk/mmv/datasets/deap/',
            'description': (
                '32 participants watched 40 music videos. '
                'EEG recorded with 32 channels. '
                'Annotated with valence, arousal, dominance, liking.'
            )
        }


class SEEDLoader(BaseDatasetLoader):
    """
    SEED Dataset 加载器
    
    数据集信息:
    - 15 名被试
    - 62 通道 EEG
    - 1000Hz 采样率
    - 3 个 sessions
    - 情绪类别：positive, neutral, negative
    
    下载链接:
    https://bcmi.sjtu.edu.cn/~seed/
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.sample_rate = 1000
        self.n_channels = 62
        self.n_sessions = 3
    
    def get_subjects(self) -> List[int]:
        """获取被试 ID 列表"""
        subjects = []
        for i in range(1, 16):
            subject_dir = self.data_dir / f"{i:02d}"
            if subject_dir.exists():
                subjects.append(i)
        return subjects
    
    def load_subject(
        self,
        subject_id: int,
        session: int = 1
    ) -> Tuple[np.ndarray, Dict]:
        """
        加载被试数据
        
        Args:
            subject_id: 被试 ID (1-15)
            session: session ID (1-3)
            
        Returns:
            (eeg_data, metadata)
        """
        subject_dir = self.data_dir / f"{subject_id:02d}"
        
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject {subject_id} not found")
        
        # 查找 Matlab 文件
        mat_files = list(subject_dir.glob(f"sess*{session}*.mat"))
        
        if not mat_files:
            raise FileNotFoundError(
                f"No session {session} data found for subject {subject_id}"
            )
        
        try:
            from scipy.io import loadmat
            
            all_trials = []
            all_labels = []
            
            for mat_file in mat_files:
                mat_data = loadmat(mat_file)
                
                # SEED 数据结构
                if 'DE_feature' in mat_data:
                    # 已经提取的特征
                    trials = mat_data['DE_feature']
                else:
                    # 原始 EEG 数据
                    trials = mat_data.get('data', mat_data.get('eegData'))
                
                if trials is not None:
                    all_trials.append(trials)
                
                # 标签
                labels = mat_data.get('labels', mat_data.get('label'))
                if labels is not None:
                    all_labels.append(labels.flatten())
            
            if not all_trials:
                raise ValueError("No data found")
            
            eeg_data = np.concatenate(all_trials, axis=0)
            labels = np.concatenate(all_labels) if all_labels else None
            
            metadata = {
                'subject_id': subject_id,
                'dataset': 'SEED',
                'session': session,
                'sample_rate': self.sample_rate,
                'n_channels': self.n_channels,
                'n_trials': len(eeg_data),
                'labels': labels,
                'emotion_classes': {
                    1: 'negative',
                    2: 'neutral',
                    3: 'positive'
                }
            }
            
            logger.info(
                f"Loaded SEED subject {subject_id}, session {session}: "
                f"shape: {eeg_data.shape}"
            )
            
            return eeg_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load SEED subject {subject_id}: {e}")
            raise
    
    def get_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'name': 'SEED',
            'n_subjects': 15,
            'n_channels': 62,
            'sample_rate': 1000,
            'n_sessions': 3,
            'emotion_classes': ['negative', 'neutral', 'positive'],
            'url': 'https://bcmi.sjtu.edu.cn/~seed/',
            'description': (
                '15 participants watched emotional film clips. '
                'EEG recorded with 62 channels. '
                'Three sessions conducted on different days.'
            )
        }


class DatasetManager:
    """
    数据集管理器
    
    统一管理多个 EEG 数据集的加载和处理
    """
    
    def __init__(self, datasets_config: Dict[str, str]):
        """
        初始化数据集管理器
        
        Args:
            datasets_config: 数据集配置 {name: data_dir}
        """
        self.datasets = {}
        self.loaders = {}
        
        for name, data_dir in datasets_config.items():
            loader = self._create_loader(name, data_dir)
            if loader:
                self.loaders[name] = loader
                self.datasets[name] = loader.get_info()
        
        logger.info(f"Initialized {len(self.loaders)} dataset loaders")
    
    def _create_loader(self, name: str, data_dir: str) -> Optional[BaseDatasetLoader]:
        """创建数据集加载器"""
        name_lower = name.lower()
        
        if 'physionet' in name_lower:
            return PhysioNetLoader(data_dir)
        elif 'deap' in name_lower:
            return DEAPLoader(data_dir)
        elif 'seed' in name_lower:
            return SEEDLoader(data_dir)
        else:
            logger.warning(f"Unknown dataset: {name}")
            return None
    
    def list_datasets(self) -> List[str]:
        """列出所有可用数据集"""
        return list(self.loaders.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name not in self.loaders:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        return self.loaders[dataset_name].get_info()
    
    def load_data(
        self,
        dataset_name: str,
        subject_id: int,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        加载数据
        
        Args:
            dataset_name: 数据集名称
            subject_id: 被试 ID
            **kwargs: 其他参数（如 session）
            
        Returns:
            (eeg_data, metadata)
        """
        if dataset_name not in self.loaders:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        loader = self.loaders[dataset_name]
        return loader.load_subject(subject_id, **kwargs)
    
    def export_metadata(self, output_dir: str):
        """导出所有数据集的元数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_path / "datasets_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(self.datasets, f, indent=2)
        
        logger.info(f"Exported metadata to {metadata_file}")


# 便捷函数

def create_dataset_manager(
    physionet_dir: Optional[str] = None,
    deap_dir: Optional[str] = None,
    seed_dir: Optional[str] = None
) -> DatasetManager:
    """
    创建数据集管理器
    
    Args:
        physionet_dir: PhysioNet 数据集目录
        deap_dir: DEAP 数据集目录
        seed_dir: SEED 数据集目录
        
    Returns:
        DatasetManager 实例
    """
    config = {}
    
    if physionet_dir:
        config['physionet'] = physionet_dir
    if deap_dir:
        config['deap'] = deap_dir
    if seed_dir:
        config['seed'] = seed_dir
    
    return DatasetManager(config)


if __name__ == "__main__":
    # 示例用法
    print("=== EEG Dataset Loaders ===\n")
    
    # 创建管理器（需要实际数据目录）
    # manager = create_dataset_manager(
    #     physionet_dir="/path/to/physionet",
    #     deap_dir="/path/to/deap",
    #     seed_dir="/path/to/seed"
    # )
    
    # 显示支持的数据集
    print("Supported datasets:")
    print("1. PhysioNet EEG Motor Movement/Imagery")
    print("   - 109 subjects, 64 channels, 160Hz")
    print("   - Motor/imagery tasks")
    print()
    print("2. DEAP")
    print("   - 32 subjects, 32 channels, 128Hz")
    print("   - 40 trials (music videos)")
    print("   - Emotion annotations")
    print()
    print("3. SEED")
    print("   - 15 subjects, 62 channels, 1000Hz")
    print("   - 3 sessions")
    print("   - Emotional film clips")
    print()
    print("Download links:")
    print("- PhysioNet: https://physionet.org/content/eegmmidb/1.0.0/")
    print("- DEAP: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/")
    print("- SEED: https://bcmi.sjtu.edu.cn/~seed/")
