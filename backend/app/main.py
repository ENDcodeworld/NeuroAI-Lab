"""
FastAPI Main Application
NeuroAI-Lab Backend API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .eeg_processor import EEGProcessor, process_eeg_file
from .api import eeg, users, training

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("🚀 NeuroAI-Lab API starting...")
    yield
    # 关闭时
    logger.info("👋 NeuroAI-Lab API shutting down...")


app = FastAPI(
    title="NeuroAI-Lab API",
    description="脑机接口数据分析服务 API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 路由注册
app.include_router(eeg.router, prefix="/api/v1/eeg", tags=["EEG 分析"])
app.include_router(users.router, prefix="/api/v1/users", tags=["用户"])
app.include_router(training.router, prefix="/api/v1/training", tags=["训练"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Welcome to NeuroAI-Lab API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "neuroai-api"
    }


@app.get("/api/v1/status")
async def status():
    """服务状态"""
    return {
        "service": "NeuroAI-Lab API",
        "version": "0.1.0",
        "status": "running"
    }


# 测试 EEG 处理
@app.post("/api/v1/test/eeg")
async def test_eeg_processing():
    """测试 EEG 处理（使用模拟数据）"""
    import numpy as np
    
    # 生成模拟 EEG 数据
    processor = EEGProcessor(sample_rate=256)
    t = np.linspace(0, 60, 256 * 60)  # 60 秒
    
    # 模拟 14 通道 EEG 数据
    data = np.zeros((14, len(t)))
    for i in range(14):
        # α波 (10Hz) + β波 (20Hz) + 噪声
        data[i] = (
            np.sin(2 * np.pi * 10 * t) * 20 +  # α
            np.sin(2 * np.pi * 20 * t) * 10 +  # β
            np.random.randn(len(t)) * 5  # 噪声
        )
    
    # 分析
    result = processor.analyze(data)
    
    return {
        "message": "EEG processing test successful",
        "data": result
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
