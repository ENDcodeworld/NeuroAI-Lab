"""
User API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
import uuid
from datetime import datetime

router = APIRouter()

# 模拟用户存储
users_db = {}


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    nickname: Optional[str] = None


class UserResponse(BaseModel):
    user_id: str
    email: str
    nickname: Optional[str]
    created_at: datetime


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate):
    """用户注册"""
    if user.email in [u["email"] for u in users_db.values()]:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    user_data = {
        "user_id": user_id,
        "email": user.email,
        "password_hash": f"hashed_{user.password}",  # 实际应使用 bcrypt
        "nickname": user.nickname,
        "created_at": datetime.now(),
    }
    
    users_db[user_id] = user_data
    
    return UserResponse(
        user_id=user_id,
        email=user.email,
        nickname=user.nickname,
        created_at=user_data["created_at"]
    )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """用户登录"""
    # 查找用户
    user = None
    for u in users_db.values():
        if u["email"] == credentials.email:
            user = u
            break
    
    if not user or user["password_hash"] != f"hashed_{credentials.password}":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # 生成 token（实际应使用 JWT）
    access_token = str(uuid.uuid4())
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            user_id=user["user_id"],
            email=user["email"],
            nickname=user["nickname"],
            created_at=user["created_at"]
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user():
    """获取当前用户信息"""
    # 实际应从 token 中解析用户 ID
    if not users_db:
        raise HTTPException(status_code=404, detail="No users found")
    
    # 返回第一个用户（测试用）
    user = list(users_db.values())[0]
    return UserResponse(
        user_id=user["user_id"],
        email=user["email"],
        nickname=user["nickname"],
        created_at=user["created_at"]
    )
