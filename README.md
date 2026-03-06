# NeuroAI-Lab 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-red.svg)](https://pytorch.org/)
[![Status: MVP](https://img.shields.io/badge/status-MVP-green.svg)]()
[![GitHub Stars](https://img.shields.io/github/stars/NeuroAI-Lab/neuroai-lab.svg)](https://github.com/NeuroAI-Lab/neuroai-lab/stargazers)
[![Issues](https://img.shields.io/github/issues/NeuroAI-Lab/neuroai-lab.svg)](https://github.com/NeuroAI-Lab/neuroai-lab/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

<div align="center">

**🧠 脑机接口 × 人工智能 创新实验室 | BCI × AI Innovation Lab**

EEG 数据分析工具 | 神经反馈训练 App | BCI 技术追踪平台 — 探索大脑奥秘，赋能人类潜能

[🚀 快速开始](#-快速开始) · [📚 文档](#-文档) · [✨ 功能特性](#-功能特性) · [🤝 贡献指南](#-贡献指南) · [💬 社区](#-社区)

![NeuroAI Demo](./docs/assets/demo.png)
*图：NeuroAI-Lab 脑电波分析界面*

</div>

---

## 🌟 项目简介

NeuroAI-Lab 是一个专注于脑机接口（BCI）与人工智能融合的创新项目，致力于开发专业的 EEG 数据分析工具、神经反馈训练应用和 BCI 技术追踪平台，让脑科学研究更普惠。

### 核心价值

| 痛点 | NeuroAI-Lab 解决方案 |
|------|---------------------|
| 🔬 EEG 数据分析复杂 | 一键式自动化分析流程 |
| 🎯 神经反馈训练门槛高 | 游戏化训练 + 成就系统 |
| 📊 BCI 行业信息分散 | 全球产业数据聚合分析 |

---

## ✨ 功能特性

### 核心能力

| 功能 | 描述 | 状态 |
|------|------|------|
| 🚀 **实时 EEG 处理** | <100ms 延迟的实时信号分析 | 🚧 开发中 |
| 🤖 **AI 驱动分析** | CNN-LSTM-Attention 深度学习模型 | 🚧 开发中 |
| 📊 **多数据集支持** | PhysioNet, DEAP, SEED 等公开数据集 | ✅ 已完成 |
| 🎮 **游戏化训练** | 神经反馈训练 + 成就系统 | 📋 规划中 |
| 🔌 **RESTful API** | 完整的 API 接口 + WebSocket 实时通信 | 🚧 开发中 |
| 📱 **跨平台** | Web + Mobile + Desktop 全支持 | 📋 规划中 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (可选)
- PostgreSQL 14+ (可选，SQLite 可用于开发)
- Redis 7+ (可选，用于缓存)

### 5 分钟快速体验

```bash
# 1. 克隆项目
git clone https://github.com/NeuroAI-Lab/neuroai-lab.git
cd neuroai-lab

# 2. 安装后端依赖
cd backend
pip install -r requirements.txt

# 3. 启动 API 服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. 测试 EEG 处理（使用模拟数据）
curl -X POST http://localhost:8000/api/v1/test/eeg
```

**响应示例:**
```json
{
  "message": "EEG processing test successful",
  "data": {
    "attention_score": 75.3,
    "relaxation_score": 62.1,
    "valence": 0.45,
    "arousal": 0.72
  }
}
```

### Docker 部署

```bash
# 使用 Docker Compose（推荐）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

访问地址：
- 🌐 前端：http://localhost:3000
- 📡 API 文档：http://localhost:8000/docs
- 💾 数据库：localhost:5432

---

## 📖 使用示例

### EEG 数据分析

```python
from app.eeg_processor import EEGProcessor, process_eeg_file

# 方法 1: 处理文件
result = process_eeg_file('subject_001.edf')
print(f"专注度：{result['attention_score']:.1f}/100")
print(f"放松度：{result['relaxation_score']:.1f}/100")

# 方法 2: 实时处理
processor = EEGProcessor(sample_rate=256)

# 从设备获取数据
data = processor.load_edf('subject.edf')

# 预处理（带 ICA 伪迹去除）
cleaned = processor.preprocess(data, method='ica')

# 特征提取
features = processor.extract_features(cleaned)
print(f"频段功率：{features['band_power']}")
print(f"左右脑不对称性：{features['asymmetry']}")

# AI 分析
result = processor.analyze(cleaned)
print(f"认知负荷：{result['cognitive_load']:.1f}/100")
```

### 神经反馈训练

```python
from app.services.neurofeedback import NeurofeedbackEngine, TrainingProgram

# 获取训练课程配置
config = TrainingProgram.get_program('focus_basic')

# 创建训练引擎
engine = NeurofeedbackEngine(config=config)

# 设置反馈回调
def on_feedback(success, power):
    if success:
        print(f"🎉 达标！功率：{power:.3f}")
    else:
        print(f"💪 继续！功率：{power:.3f}")

engine.set_feedback_callback(on_feedback)

# 开始训练
import asyncio

async def train():
    await engine.start_session()
    
    # 模拟 EEG 数据流
    for i in range(100):
        eeg_data = get_eeg_from_device()  # 从设备获取
        await engine.process_eeg_data(eeg_data)
        await asyncio.sleep(0.1)
    
    await engine.end_session()
    
    # 查看结果
    metrics = engine.get_current_metrics()
    print(f"达标率：{metrics['success_rate']:.2%}")

asyncio.run(train())
```

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      客户端层                                │
│         Web 应用 │ 移动应用 │ 桌面应用 │ API 客户端          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API 网关                                  │
│              认证 │ 限流 │ 路由                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   微服务层                                   │
│  EEG 分析服务 │ 训练服务 │ 用户服务 │ 数据服务 │ AI 服务     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     数据层                                   │
│     PostgreSQL │ Redis │ MinIO/S3 │ EEG 数据库              │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **前端** | Next.js 14 + React 18 + TypeScript | 现代化 Web 界面 |
| **可视化** | D3.js + Three.js | 3D 脑图可视化 |
| **后端** | Python 3.10 + FastAPI | 高性能异步 API |
| **深度学习** | PyTorch 2.2 + ONNX Runtime | AI 模型推理 |
| **信号处理** | MNE-Python + NumPy + SciPy | EEG 信号处理 |
| **数据库** | PostgreSQL 15 + Redis 7 | 关系型 + 缓存 |
| **移动端** | React Native (开发中) | 跨平台移动应用 |

---

## 📚 文档

| 文档 | 说明 | 链接 |
|------|------|------|
| 📘 安装指南 | 详细安装步骤 | [查看](docs/installation.md) |
| 📗 快速入门 | 5 分钟上手教程 | [查看](docs/quickstart.md) |
| 📙 API 参考 | 完整 API 文档 | [查看](docs/API_REFERENCE.md) |
| 📕 示例代码 | 实用示例集合 | [查看](examples/) |
| 📒 贡献指南 | 如何贡献代码 | [查看](CONTRIBUTING.md) |

---

## 🗺️ 路线图

<div align="center">

| 时间 | 里程碑 | 状态 |
|------|--------|------|
| 2026 Q1 | EEG 信号处理模块 + 数据集集成 | ✅ 已完成 |
| 2026 Q2 | 前端界面 + 移动端 App | 🚧 进行中 |
| 2026 Q3 | 神经反馈 App 上线 + 游戏化系统 | 📋 规划中 |
| 2026 Q4 | BCI 技术追踪平台 + 企业版 API | 📋 规划中 |

</div>

详细路线图请查看 [ROADMAP.md](docs/ROADMAP.md)

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. 🍴 **Fork 仓库** - 创建你自己的 fork
2. 🌿 **创建分支** - `git checkout -b feature/amazing-feature`
3. 💻 **开发** - 编写代码和测试
4. ✅ **测试** - 确保所有测试通过
5. 📤 **提交 PR** - 描述你的改动

### 开发环境设置

```bash
# Fork & Clone
git clone https://github.com/YOUR_USERNAME/neuroai-lab.git
cd neuroai-lab

# 安装依赖
cd backend && pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 代码格式化
black app/
ruff check app/
```

### 代码规范

- **Python:** 遵循 PEP 8 + Black 格式化
- **TypeScript:** 遵循 ESLint + Prettier
- **提交信息:** 遵循 Conventional Commits 规范

详见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行 EEG 处理专项测试
pytest tests/test_eeg_processor.py -v
pytest tests/test_neurofeedback.py -v

# 带覆盖率报告
pytest --cov=app --cov-report=html
```

---

## 📊 项目统计

[![Star History](https://api.star-history.com/svg?repos=NeuroAI-Lab/neuroai-lab&type=Date)](https://star-history.com/#NeuroAI-Lab/neuroai-lab&Date)

| 指标 | 数据 |
|------|------|
| ⭐ Stars | 0 |
| 🍴 Forks | 0 |
| 🐛 Issues | 0 |
| 📦 Downloads | 0 |

---

## 💬 社区

### 联系方式

| 平台 | 链接 |
|------|------|
| 🌐 官网 | https://neuroai-lab.com (即将上线) |
| 📧 邮箱 | contact@neuroai-lab.com |
| 💬 Discord | [加入社区](https://discord.gg/neuroai) |
| 🐦 Twitter | [@NeuroAI_Lab](https://twitter.com/NeuroAI_Lab) |
| 📱 微信 | NeuroAI 实验室公众号 |
| 📺 B 站 | @NeuroAI-Lab |
| 📖 知乎 | @NeuroAI-Lab |

### 加入讨论

- 💬 **Discord 服务器**: [点击加入](https://discord.gg/neuroai)
- 📱 **微信群**: 添加小助手微信 `neuroai_assistant` 邀请入群
- 🐦 **Twitter**: [@NeuroAI_Lab](https://twitter.com/NeuroAI_Lab)

---

## 💰 赞助商

NeuroAI-Lab 是开源项目，感谢以下赞助商的支持：

<div align="center">

| 赞助商等级 | 赞助商 | 链接 |
|-----------|--------|------|
| 🏆 **金牌赞助商** | [虚位以待] | [成为赞助商](mailto:sponsor@neuroai-lab.com) |
| 🥈 **银牌赞助商** | [虚位以待] | [成为赞助商](mailto:sponsor@neuroai-lab.com) |
| 🥉 **铜牌赞助商** | [虚位以待] | [成为赞助商](mailto:sponsor@neuroai-lab.com) |

</div>

### 赞助方式

我们接受以下形式的赞助：

- 💰 **资金赞助** - 支持项目持续开发
- 🖥️ **云服务资源** - 服务器、存储、CDN
- 🎯 **推广支持** - 社交媒体分享、技术文章
- 👨‍💻 **人才赞助** - 开发者贡献时间

[👉 立即赞助](https://github.com/sponsors/NeuroAI-Lab) | [📧 联系合作](mailto:sponsor@neuroai-lab.com)

---

## 🙏 致谢

感谢以下优秀的开源项目：

- [MNE-Python](https://mne.tools/) - EEG/MEG 信号处理
- [EEGLab](https://eeglab.org/) - 脑电分析工具
- [PhysioNet](https://physionet.org/) - 生理信号数据集
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 API 框架

---

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件

---

## 👥 团队

- **创始人**: 志哥
- **核心团队**: NeuroAI 开发团队
- **贡献者**: [查看贡献者列表](https://github.com/NeuroAI-Lab/neuroai-lab/graphs/contributors)

---

<div align="center">

### ⭐ 喜欢这个项目吗？

如果这个项目对你有帮助，请给我们一个 **Star** 支持！你的支持是我们持续开发的动力！

[![Star](https://img.shields.io/github/stars/NeuroAI-Lab/neuroai-lab?style=social)](https://github.com/NeuroAI-Lab/neuroai-lab)

---

**Made with ❤️ by NeuroAI-Lab Team**

🧠 *探索大脑奥秘，赋能人类潜能*

[⬆ 返回顶部](#neuroai-lab-)

</div>

---

## 🔍 SEO 关键词

NeuroAI-Lab, 脑机接口，EEG 分析，神经反馈，AI 脑科学，脑电波，open source, AI, machine learning, deep learning, BCI, neuroscience, EEG
