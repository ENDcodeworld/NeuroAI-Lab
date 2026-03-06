"""
Training API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
from datetime import datetime

router = APIRouter()

# 模拟训练课程
training_programs = [
    {
        "program_id": "focus-basic",
        "name": "专注力基础训练",
        "description": "提升日常专注力，适合初学者",
        "difficulty": "beginner",
        "duration_minutes": 15,
        "target_band": "beta"
    },
    {
        "program_id": "focus-advanced",
        "name": "专注力进阶训练",
        "description": "深度专注力训练，提升工作效率",
        "difficulty": "intermediate",
        "duration_minutes": 25,
        "target_band": "beta"
    },
    {
        "program_id": "meditation",
        "name": "冥想放松训练",
        "description": "通过α波训练达到深度放松",
        "difficulty": "beginner",
        "duration_minutes": 20,
        "target_band": "alpha"
    },
    {
        "program_id": "sleep",
        "name": "睡眠改善训练",
        "description": "改善睡眠质量，缩短入睡时间",
        "difficulty": "beginner",
        "duration_minutes": 30,
        "target_band": "theta"
    },
]

# 模拟训练记录
training_records = {}


class TrainingSession(BaseModel):
    program_id: str
    duration_minutes: Optional[int] = None


class SessionResponse(BaseModel):
    session_id: str
    program_id: str
    status: str
    config: Dict


@router.get("/programs", response_model=List[Dict])
async def list_programs():
    """获取训练课程列表"""
    return training_programs


@router.post("/sessions", response_model=SessionResponse)
async def start_session(session: TrainingSession):
    """开始训练会话"""
    # 验证课程
    program = next((p for p in training_programs if p["program_id"] == session.program_id), None)
    if not program:
        raise HTTPException(status_code=404, detail="Program not found")
    
    session_id = str(uuid.uuid4())
    duration = session.duration_minutes or program["duration_minutes"]
    
    # 创建会话
    training_records[session_id] = {
        "session_id": session_id,
        "program_id": session.program_id,
        "status": "active",
        "started_at": datetime.now(),
        "duration_minutes": duration,
        "metrics": None
    }
    
    return SessionResponse(
        session_id=session_id,
        program_id=session.program_id,
        status="active",
        config={
            "target_band": program["target_band"],
            "duration_minutes": duration,
            "threshold": 0.7
        }
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取训练会话详情"""
    if session_id not in training_records:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return training_records[session_id]


@router.post("/sessions/{session_id}/complete")
async def complete_session(session_id: str, metrics: Dict):
    """完成训练会话"""
    if session_id not in training_records:
        raise HTTPException(status_code=404, detail="Session not found")
    
    training_records[session_id]["status"] = "completed"
    training_records[session_id]["completed_at"] = datetime.now()
    training_records[session_id]["metrics"] = metrics
    
    # 计算经验值
    exp_points = int(metrics.get("performance_score", 0) * 10)
    
    return {
        "session_id": session_id,
        "status": "completed",
        "experience_points": exp_points,
        "achievements": []
    }


@router.get("/history")
async def get_training_history():
    """获取训练历史"""
    return list(training_records.values())
