"""
EEG Analysis API Routes
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# 模拟任务存储（生产环境应使用 Redis/数据库）
tasks: Dict[str, Dict] = {}


class AnalysisRequest(BaseModel):
    """分析请求"""
    device_type: Optional[str] = "muse"
    session_config: Optional[Dict] = None


class AnalysisResponse(BaseModel):
    """分析响应"""
    task_id: str
    status: str
    result: Optional[Dict] = None


@router.post("/upload", response_model=AnalysisResponse)
async def upload_eeg(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="EEG 数据文件 (EDF/BDF/CSV)")
):
    """
    上传 EEG 数据文件进行分析
    
    - **file**: EEG 数据文件
    """
    task_id = str(uuid.uuid4())
    
    # 保存文件（临时）
    file_path = f"/tmp/{task_id}_{file.filename}"
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 创建任务
        tasks[task_id] = {
            "status": "processing",
            "file_path": file_path,
            "progress": 0,
            "result": None
        }
        
        # 后台处理
        background_tasks.add_task(process_eeg_task, task_id, file_path)
        
        return AnalysisResponse(
            task_id=task_id,
            status="processing"
        )
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_eeg_task(task_id: str, file_path: str):
    """后台处理 EEG 分析任务"""
    try:
        from app.eeg_processor import process_eeg_file
        
        # 更新进度
        tasks[task_id]["progress"] = 25
        
        # 处理文件
        result = process_eeg_file(file_path)
        
        # 更新进度
        tasks[task_id]["progress"] = 100
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        
        logger.info(f"Task {task_id} completed")
    
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


@router.get("/analysis/{task_id}", response_model=AnalysisResponse)
async def get_analysis_result(task_id: str):
    """
    查询分析任务进度/结果
    
    - **task_id**: 任务 ID
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return AnalysisResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result")
    )


@router.post("/realtime/start")
async def start_realtime_session(request: AnalysisRequest):
    """
    开始实时分析会话
    
    返回 WebSocket 连接 URL
    """
    session_id = str(uuid.uuid4())
    
    return {
        "session_id": session_id,
        "websocket_url": f"ws://localhost:8000/api/v1/eeg/realtime/{session_id}",
        "config": {
            "device_type": request.device_type,
            "sample_rate": 256,
            "channels": 14
        }
    }


@router.get("/devices")
async def list_supported_devices():
    """获取支持的设备列表"""
    return {
        "devices": [
            {
                "type": "muse",
                "name": "Muse 2/S",
                "channels": 4,
                "sample_rate": 256,
                "connection": "Bluetooth"
            },
            {
                "type": "emotiv",
                "name": "Emotiv EPOC X",
                "channels": 14,
                "sample_rate": 256,
                "connection": "Bluetooth/USB"
            },
            {
                "type": "openbci",
                "name": "OpenBCI Ganglion",
                "channels": 4,
                "sample_rate": 200,
                "connection": "Bluetooth"
            }
        ]
    }
