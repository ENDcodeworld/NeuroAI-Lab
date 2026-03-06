"""
Neurofeedback Training Engine
神经反馈训练引擎

功能:
- 实时 EEG 分析
- 反馈阈值控制
- 训练课程设计
- 游戏化反馈
- 进度追踪

作者：NeuroAI-Lab Team
版本：1.0.0
"""

import numpy as np
import asyncio
import logging
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型"""
    VISUAL = 'visual'      # 视觉反馈
    AUDIO = 'audio'        # 听觉反馈
    HAPTIC = 'haptic'      # 触觉反馈
    GAME = 'game'          # 游戏化反馈


class TrainingState(Enum):
    """训练状态"""
    IDLE = 'idle'
    CONNECTING = 'connecting'
    CALIBRATING = 'calibrating'
    TRAINING = 'training'
    PAUSED = 'paused'
    COMPLETED = 'completed'


@dataclass
class TrainingConfig:
    """训练配置"""
    # 目标频段
    target_band: str = 'alpha'
    
    # 阈值设置
    threshold: float = 0.7  # 达标阈值 (0-1)
    threshold_auto_adjust: bool = True
    
    # 训练参数
    session_duration: int = 20  # 分钟
    block_duration: int = 3     # 每个训练块时长（分钟）
    rest_duration: int = 1      # 休息时长（分钟）
    
    # 反馈设置
    feedback_type: FeedbackType = FeedbackType.VISUAL
    feedback_delay: float = 0.1  # 反馈延迟（秒）
    
    # 难度曲线
    difficulty_curve: str = 'adaptive'  # 'fixed', 'adaptive', 'progressive'
    
    # 奖励设置
    reward_interval: int = 5  # 达标多少秒给予奖励


@dataclass
class TrainingMetrics:
    """训练指标"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # 基础指标
    total_duration: float = 0.0  # 总时长（秒）
    training_duration: float = 0.0  # 有效训练时长
    
    # 表现指标
    success_rate: float = 0.0  # 达标率
    average_band_power: float = 0.0  # 平均频段功率
    
    # 时间序列数据
    band_power_history: List[float] = field(default_factory=list)
    success_history: List[bool] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    # 统计
    total_rewards: int = 0
    max_streak: int = 0
    current_streak: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': self.total_duration,
            'training_duration': self.training_duration,
            'success_rate': self.success_rate,
            'average_band_power': self.average_band_power,
            'total_rewards': self.total_rewards,
            'max_streak': self.max_streak,
        }


class NeurofeedbackEngine:
    """
    神经反馈训练引擎
    
    实时分析 EEG 信号，提供神经反馈训练
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        eeg_processor=None
    ):
        """
        初始化训练引擎
        
        Args:
            config: 训练配置
            eeg_processor: EEG 处理器实例
        """
        self.config = config or TrainingConfig()
        self.eeg_processor = eeg_processor
        
        # 状态
        self.state = TrainingState.IDLE
        self.session_id = None
        self.metrics = None
        
        # 实时数据
        self.current_band_power = 0.0
        self.is_success = False
        self.session_start = None
        
        # 回调函数
        self.feedback_callback: Optional[Callable] = None
        self.progress_callback: Optional[Callable] = None
        
        # 校准数据
        self.baseline_band_power = None
        self.baseline_std = None
        
        logger.info("NeurofeedbackEngine initialized")
    
    def set_feedback_callback(self, callback: Callable):
        """
        设置反馈回调
        
        Args:
            callback: 回调函数 (success: bool, power: float)
        """
        self.feedback_callback = callback
        logger.info("Feedback callback set")
    
    def set_progress_callback(self, callback: Callable):
        """
        设置进度回调
        
        Args:
            callback: 回调函数 (metrics: TrainingMetrics)
        """
        self.progress_callback = callback
    
    async def start_session(self, session_id: Optional[str] = None):
        """
        开始训练会话
        
        Args:
            session_id: 会话 ID
        """
        if self.state != TrainingState.IDLE:
            logger.warning(f"Cannot start session: current state is {self.state}")
            return
        
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state = TrainingState.CONNECTING
        self.session_start = datetime.now()
        
        # 初始化指标
        self.metrics = TrainingMetrics(
            session_id=self.session_id,
            start_time=self.session_start
        )
        
        logger.info(f"Starting session: {self.session_id}")
        
        # 校准阶段
        await self._calibrate()
        
        # 开始训练
        self.state = TrainingState.TRAINING
        logger.info("Training session started")
    
    async def _calibrate(self, duration: int = 30):
        """
        校准基线
        
        Args:
            duration: 校准时长（秒）
        """
        logger.info(f"Starting calibration ({duration}s)...")
        self.state = TrainingState.CALIBRATING
        
        # 收集基线数据
        baseline_samples = []
        
        for _ in range(duration * 10):  # 100Hz 采样
            power = self._get_current_band_power()
            baseline_samples.append(power)
            await asyncio.sleep(0.1)
        
        # 计算统计量
        self.baseline_band_power = np.mean(baseline_samples)
        self.baseline_std = np.std(baseline_samples)
        
        # 设置初始阈值
        if self.config.threshold_auto_adjust:
            self.config.threshold = self.baseline_band_power + 0.5 * self.baseline_std
        
        logger.info(
            f"Calibration complete: baseline={self.baseline_band_power:.3f}, "
            f"std={self.baseline_std:.3f}, threshold={self.config.threshold:.3f}"
        )
        
        self.state = TrainingState.TRAINING
    
    def _get_current_band_power(self) -> float:
        """获取当前频段功率"""
        # 从 EEG 处理器获取实时数据
        if self.eeg_processor and hasattr(self.eeg_processor, 'get_latest_band_power'):
            return self.eeg_processor.get_latest_band_power(self.config.target_band)
        
        # 模拟数据（用于测试）
        return np.random.uniform(0.3, 0.8)
    
    async def process_eeg_data(self, eeg_data: np.ndarray):
        """
        处理实时 EEG 数据
        
        Args:
            eeg_data: EEG 数据 (channels, samples)
        """
        if self.state != TrainingState.TRAINING:
            return
        
        # 预处理
        if self.eeg_processor:
            cleaned = self.eeg_processor.preprocess(eeg_data)
            
            # 提取频段功率
            band_power = self.eeg_processor.calc_band_power(cleaned)
            self.current_band_power = np.mean(band_power[self.config.target_band])
        
        # 判断是否达标
        self.is_success = self.current_band_power > self.config.threshold
        
        # 更新指标
        self._update_metrics()
        
        # 触发反馈
        if self.feedback_callback:
            self.feedback_callback(self.is_success, self.current_band_power)
        
        # 调整阈值（自适应难度）
        if self.config.threshold_auto_adjust:
            self._adjust_threshold()
    
    def _update_metrics(self):
        """更新训练指标"""
        if not self.metrics:
            return
        
        current_time = (datetime.now() - self.session_start).total_seconds()
        
        # 记录数据
        self.metrics.band_power_history.append(self.current_band_power)
        self.metrics.success_history.append(self.is_success)
        self.metrics.timestamps.append(current_time)
        
        # 更新达标率
        if len(self.metrics.success_history) > 0:
            self.metrics.success_rate = (
                sum(self.metrics.success_history) /
                len(self.metrics.success_history)
            )
        
        # 更新连续达标
        if self.is_success:
            self.metrics.current_streak += 1
            self.metrics.max_streak = max(
                self.metrics.max_streak,
                self.metrics.current_streak
            )
            
            # 给予奖励
            if self.metrics.current_streak % self.config.reward_interval == 0:
                self.metrics.total_rewards += 1
        else:
            self.metrics.current_streak = 0
        
        # 平均频段功率
        self.metrics.average_band_power = np.mean(self.metrics.band_power_history)
        
        # 进度回调
        if self.progress_callback:
            self.progress_callback(self.metrics)
    
    def _adjust_threshold(self):
        """自适应调整阈值"""
        if self.config.difficulty_curve == 'fixed':
            return
        
        # 基于最近表现调整
        recent_success = self.metrics.success_history[-10:] if len(self.metrics.success_history) >= 10 else []
        
        if len(recent_success) > 0:
            recent_rate = sum(recent_success) / len(recent_success)
            
            if recent_rate > 0.8:
                # 表现太好，提高难度
                self.config.threshold *= 1.05
                logger.debug(f"Threshold increased to {self.config.threshold:.3f}")
            elif recent_rate < 0.3:
                # 表现太差，降低难度
                self.config.threshold *= 0.95
                logger.debug(f"Threshold decreased to {self.config.threshold:.3f}")
    
    async def pause_session(self):
        """暂停训练"""
        if self.state == TrainingState.TRAINING:
            self.state = TrainingState.PAUSED
            logger.info("Session paused")
    
    async def resume_session(self):
        """恢复训练"""
        if self.state == TrainingState.PAUSED:
            self.state = TrainingState.TRAINING
            logger.info("Session resumed")
    
    async def end_session(self):
        """结束训练会话"""
        if self.state in [TrainingState.TRAINING, TrainingState.PAUSED]:
            self.state = TrainingState.COMPLETED
            self.metrics.end_time = datetime.now()
            
            # 计算总时长
            self.metrics.total_duration = (
                self.metrics.end_time - self.metrics.start_time
            ).total_seconds()
            
            logger.info(
                f"Session completed: {self.metrics.session_id}, "
                f"duration={self.metrics.total_duration:.1f}s, "
                f"success_rate={self.metrics.success_rate:.2%}"
            )
    
    def get_current_metrics(self) -> Dict:
        """获取当前指标"""
        if not self.metrics:
            return {}
        
        return {
            'state': self.state.value,
            'band_power': self.current_band_power,
            'is_success': self.is_success,
            'threshold': self.config.threshold,
            'success_rate': self.metrics.success_rate,
            'duration': (datetime.now() - self.session_start).total_seconds() if self.session_start else 0,
        }
    
    def export_results(self, output_path: str):
        """导出训练结果"""
        if not self.metrics:
            raise ValueError("No session data to export")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 导出为 JSON
        with open(output_file, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        logger.info(f"Results exported to {output_file}")


class TrainingProgram:
    """
    训练课程管理器
    
    预定义的训练课程模板
    """
    
    @staticmethod
    def get_program(program_name: str) -> TrainingConfig:
        """
        获取训练课程配置
        
        Args:
            program_name: 课程名称
            
        Returns:
            TrainingConfig
        """
        programs = {
            'focus_basic': TrainingConfig(
                target_band='beta',
                threshold=0.6,
                session_duration=15,
                block_duration=2,
                feedback_type=FeedbackType.VISUAL,
            ),
            
            'focus_advanced': TrainingConfig(
                target_band='beta',
                threshold=0.7,
                threshold_auto_adjust=True,
                session_duration=25,
                block_duration=5,
                feedback_type=FeedbackType.GAME,
            ),
            
            'relaxation': TrainingConfig(
                target_band='alpha',
                threshold=0.65,
                session_duration=20,
                block_duration=3,
                feedback_type=FeedbackType.AUDIO,
            ),
            
            'meditation': TrainingConfig(
                target_band='theta',
                threshold=0.6,
                session_duration=30,
                block_duration=5,
                feedback_type=FeedbackType.VISUAL,
            ),
            
            'sleep_preparation': TrainingConfig(
                target_band='delta',
                threshold=0.5,
                session_duration=20,
                block_duration=4,
                feedback_type=FeedbackType.AUDIO,
            ),
        }
        
        if program_name not in programs:
            raise ValueError(f"Unknown program: {program_name}")
        
        return programs[program_name]
    
    @staticmethod
    def list_programs() -> List[Dict]:
        """列出所有训练课程"""
        return [
            {
                'name': 'focus_basic',
                'description': '基础专注力训练',
                'target': 'Beta 波增强',
                'duration': '15 分钟',
                'difficulty': '初级',
            },
            {
                'name': 'focus_advanced',
                'description': '高级专注力训练',
                'target': 'Beta 波增强（自适应难度）',
                'duration': '25 分钟',
                'difficulty': '高级',
            },
            {
                'name': 'relaxation',
                'description': '放松训练',
                'target': 'Alpha 波增强',
                'duration': '20 分钟',
                'difficulty': '初级',
            },
            {
                'name': 'meditation',
                'description': '冥想训练',
                'target': 'Theta 波增强',
                'duration': '30 分钟',
                'difficulty': '中级',
            },
            {
                'name': 'sleep_preparation',
                'description': '睡眠准备',
                'target': 'Delta 波增强',
                'duration': '20 分钟',
                'difficulty': '初级',
            },
        ]


class FeedbackRenderer:
    """
    反馈渲染器
    
    提供视觉、听觉、游戏化反馈
    """
    
    @staticmethod
    def create_visual_feedback(success: bool, power: float) -> Dict:
        """
        创建视觉反馈
        
        Args:
            success: 是否达标
            power: 当前频段功率
            
        Returns:
            反馈参数
        """
        # 颜色映射（红->黄->绿）
        if power < 0.3:
            color = '#FF4444'  # 红色
        elif power < 0.6:
            color = '#FFAA00'  # 橙色
        else:
            color = '#44FF44'  # 绿色
        
        return {
            'type': 'visual',
            'color': color,
            'brightness': min(1.0, power),
            'success': success,
            'animation': 'pulse' if success else 'steady',
        }
    
    @staticmethod
    def create_audio_feedback(success: bool, power: float) -> Dict:
        """
        创建听觉反馈
        
        Returns:
            音频参数
        """
        if success:
            return {
                'type': 'audio',
                'frequency': 440 + power * 220,  # 440-660Hz
                'volume': 0.3 + power * 0.4,
                'duration': 0.2,
                'waveform': 'sine',
            }
        else:
            return {
                'type': 'audio',
                'frequency': 220,
                'volume': 0.1,
                'duration': 0.1,
                'waveform': 'sine',
            }
    
    @staticmethod
    def create_game_feedback(success: bool, power: float, streak: int) -> Dict:
        """
        创建游戏化反馈
        
        Returns:
            游戏反馈参数
        """
        feedback = {
            'type': 'game',
            'score_delta': int(power * 100) if success else 0,
            'streak_bonus': streak * 10 if streak > 5 else 0,
            'effect': None,
        }
        
        if success:
            if streak >= 10:
                feedback['effect'] = 'fire'
            elif streak >= 5:
                feedback['effect'] = 'sparkle'
            else:
                feedback['effect'] = 'glow'
        
        return feedback


# ==================== WebSocket 实时通信 ====================

async def websocket_handler(websocket, engine: NeurofeedbackEngine):
    """
    WebSocket 处理器
    
    用于实时神经反馈
    """
    import json
    
    async def on_feedback(success: bool, power: float):
        """反馈回调"""
        feedback = FeedbackRenderer.create_visual_feedback(success, power)
        await websocket.send_json({
            'type': 'feedback',
            'data': feedback,
            'metrics': engine.get_current_metrics(),
        })
    
    engine.set_feedback_callback(on_feedback)
    
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'eeg_data':
                # 接收 EEG 数据
                eeg_data = np.array(data['data'])
                await engine.process_eeg_data(eeg_data)
            
            elif data['type'] == 'pause':
                await engine.pause_session()
            
            elif data['type'] == 'resume':
                await engine.resume_session()
            
            elif data['type'] == 'end':
                await engine.end_session()
                break
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await engine.end_session()


if __name__ == "__main__":
    # 测试训练引擎
    print("=== Neurofeedback Engine Test ===\n")
    
    # 创建引擎
    engine = NeurofeedbackEngine(
        config=TrainingConfig(
            target_band='alpha',
            session_duration=5,  # 测试用短时
        )
    )
    
    # 设置反馈回调
    def feedback_callback(success: bool, power: float):
        status = "✓" if success else "✗"
        print(f"{status} Power: {power:.3f}")
    
    engine.set_feedback_callback(feedback_callback)
    
    # 模拟训练
    async def test_training():
        await engine.start_session()
        
        # 模拟 EEG 数据
        for i in range(50):
            # 生成模拟数据
            eeg_data = np.random.randn(14, 256)
            await engine.process_eeg_data(eeg_data)
            await asyncio.sleep(0.1)
        
        await engine.end_session()
        
        # 显示结果
        metrics = engine.get_current_metrics()
        print(f"\n=== Session Results ===")
        print(f"Duration: {metrics['duration']:.1f}s")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Final Threshold: {metrics['threshold']:.3f}")
    
    # 运行测试
    asyncio.run(test_training())
