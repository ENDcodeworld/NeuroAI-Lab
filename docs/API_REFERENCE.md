# NeuroAI-Lab API 参考文档

**版本:** 1.0.0  
**最后更新:** 2026-03-06

---

## 📡 API 概览

Base URL: `http://localhost:8000/api/v1`

### 认证

大部分 API 需要 JWT Token 认证：

```http
Authorization: Bearer <your_token>
```

### 响应格式

所有响应均为 JSON 格式：

```json
{
  "success": true,
  "data": {...},
  "message": "操作成功"
}
```

错误响应：

```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "详细的错误信息"
  }
}
```

---

## 🔌 EEG 分析 API

### 1. 上传 EEG 数据

**端点:** `POST /eeg/upload`

**描述:** 上传 EEG 数据文件进行分析

**请求参数:**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| file | File | 是 | EEG 数据文件 (EDF/BDF/CSV) |
| device_type | String | 否 | 设备类型 (muse/emotiv/openbci) |

**请求示例:**

```bash
curl -X POST http://localhost:8000/api/v1/eeg/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@subject_001.edf" \
  -F "device_type=muse"
```

**响应示例:**

```json
{
  "success": true,
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "estimated_time": 30
  }
}
```

---

### 2. 查询分析结果

**端点:** `GET /eeg/analysis/{task_id}`

**描述:** 查询 EEG 分析任务的进度和结果

**路径参数:**

| 参数 | 类型 | 描述 |
|------|------|------|
| task_id | String | 任务 ID |

**响应示例:**

```json
{
  "success": true,
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "progress": 100,
    "result": {
      "attention_score": 75.3,
      "relaxation_score": 62.1,
      "valence": 0.45,
      "arousal": 0.72,
      "cognitive_load": 58.9,
      "band_power": {
        "delta": [12.5, 13.2, ...],
        "theta": [8.7, 9.1, ...],
        "alpha": [15.3, 14.8, ...],
        "beta": [10.2, 11.5, ...],
        "gamma": [5.6, 6.2, ...]
      },
      "asymmetry": {
        "alpha": 0.12,
        "beta": -0.08
      }
    }
  }
}
```

---

### 3. 开始实时分析会话

**端点:** `POST /eeg/realtime/start`

**描述:** 开始实时 EEG 分析会话

**请求参数:**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| device_type | String | 是 | 设备类型 |
| sample_rate | Integer | 否 | 采样率 (默认 256) |
| channels | Integer | 否 | 通道数 (默认 14) |

**请求示例:**

```bash
curl -X POST http://localhost:8000/api/v1/eeg/realtime/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "device_type": "muse",
    "sample_rate": 256,
    "channels": 4
  }'
```

**响应示例:**

```json
{
  "success": true,
  "data": {
    "session_id": "sess_123456",
    "websocket_url": "ws://localhost:8000/api/v1/eeg/realtime/sess_123456",
    "config": {
      "device_type": "muse",
      "sample_rate": 256,
      "channels": 4
    }
  }
}
```

---

### 4. 实时数据流 (WebSocket)

**端点:** `WS /eeg/realtime/{session_id}`

**描述:** 通过 WebSocket 接收实时分析结果

**连接示例:**

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/eeg/realtime/sess_123456');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Attention:', data.metrics.attention_score);
};

// 发送 EEG 数据
function sendEEGData(eegData) {
  ws.send(JSON.stringify({
    type: 'eeg_data',
    data: eegData
  }));
}
```

**消息格式:**

```json
{
  "type": "feedback",
  "data": {
    "color": "#44FF44",
    "brightness": 0.85,
    "success": true
  },
  "metrics": {
    "attention_score": 78.5,
    "band_power": 0.85,
    "is_success": true,
    "threshold": 0.70
  }
}
```

---

### 5. 获取支持的设备列表

**端点:** `GET /eeg/devices`

**描述:** 获取所有支持的 EEG 设备信息

**响应示例:**

```json
{
  "success": true,
  "data": {
    "devices": [
      {
        "type": "muse",
        "name": "Muse 2/S",
        "channels": 4,
        "sample_rate": 256,
        "connection": "Bluetooth",
        "price_range": "$200-250"
      },
      {
        "type": "emotiv",
        "name": "Emotiv EPOC X",
        "channels": 14,
        "sample_rate": 256,
        "connection": "Bluetooth/USB",
        "price_range": "$800-1000"
      },
      {
        "type": "openbci",
        "name": "OpenBCI Ganglion",
        "channels": 4,
        "sample_rate": 200,
        "connection": "Bluetooth",
        "price_range": "$500-600"
      }
    ]
  }
}
```

---

## 🎯 训练 API

### 1. 获取训练课程列表

**端点:** `GET /training/programs`

**描述:** 获取所有可用的神经反馈训练课程

**响应示例:**

```json
{
  "success": true,
  "data": [
    {
      "program_id": "focus_basic",
      "name": "基础专注力训练",
      "description": "适合初学者的专注力提升训练",
      "target_band": "beta",
      "duration_minutes": 15,
      "difficulty": "beginner",
      "benefits": ["提升注意力", "改善工作效率"]
    },
    {
      "program_id": "relaxation",
      "name": "放松训练",
      "description": "通过 Alpha 波训练达到深度放松",
      "target_band": "alpha",
      "duration_minutes": 20,
      "difficulty": "beginner",
      "benefits": ["减轻压力", "改善睡眠"]
    }
  ]
}
```

---

### 2. 开始训练会话

**端点:** `POST /training/sessions`

**描述:** 开始一个新的训练会话

**请求参数:**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| program_id | String | 是 | 训练课程 ID |
| duration_minutes | Integer | 否 | 训练时长（分钟） |

**请求示例:**

```bash
curl -X POST http://localhost:8000/api/v1/training/sessions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "program_id": "focus_basic",
    "duration_minutes": 15
  }'
```

**响应示例:**

```json
{
  "success": true,
  "data": {
    "session_id": "train_789",
    "program_id": "focus_basic",
    "status": "ready",
    "config": {
      "target_band": "beta",
      "threshold": 0.6,
      "duration_minutes": 15
    },
    "websocket_url": "ws://localhost:8000/api/v1/training/sessions/train_789"
  }
}
```

---

### 3. 获取训练会话详情

**端点:** `GET /training/sessions/{session_id}`

**描述:** 获取训练会话的详细信息和进度

**响应示例:**

```json
{
  "success": true,
  "data": {
    "session_id": "train_789",
    "program_id": "focus_basic",
    "status": "in_progress",
    "progress": {
      "elapsed_seconds": 420,
      "total_seconds": 900,
      "percent_complete": 46.7
    },
    "metrics": {
      "success_rate": 0.72,
      "average_band_power": 0.65,
      "current_streak": 8,
      "total_rewards": 12
    },
    "start_time": "2026-03-06T14:30:00Z"
  }
}
```

---

### 4. 完成训练会话

**端点:** `POST /training/sessions/{session_id}/complete`

**描述:** 完成训练会话并保存结果

**请求参数:**

| 参数 | 类型 | 描述 |
|------|------|------|
| final_metrics | Object | 最终指标 |

**响应示例:**

```json
{
  "success": true,
  "data": {
    "session_id": "train_789",
    "completed": true,
    "summary": {
      "duration_minutes": 15.2,
      "success_rate": 0.68,
      "experience_points": 150,
      "achievements": ["首次训练", "专注达人"]
    },
    "next_recommended": "focus_advanced"
  }
}
```

---

### 5. 获取训练历史

**端点:** `GET /training/history`

**描述:** 获取用户的训练历史记录

**查询参数:**

| 参数 | 类型 | 描述 |
|------|------|------|
| limit | Integer | 返回数量限制（默认 20） |
| offset | Integer | 偏移量 |
| program_id | String | 按课程筛选 |

**响应示例:**

```json
{
  "success": true,
  "data": {
    "total": 45,
    "sessions": [
      {
        "session_id": "train_789",
        "program_id": "focus_basic",
        "date": "2026-03-06",
        "duration_minutes": 15.2,
        "success_rate": 0.68,
        "score": 85
      },
      {
        "session_id": "train_756",
        "program_id": "relaxation",
        "date": "2026-03-05",
        "duration_minutes": 20.0,
        "success_rate": 0.75,
        "score": 92
      }
    ]
  }
}
```

---

## 👤 用户 API

### 1. 用户注册

**端点:** `POST /auth/register`

**请求参数:**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| email | String | 是 | 邮箱地址 |
| password | String | 是 | 密码（最少 8 位） |
| nickname | String | 否 | 昵称 |

**响应示例:**

```json
{
  "success": true,
  "data": {
    "user_id": "user_123",
    "email": "user@example.com",
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_at": "2026-03-07T14:30:00Z"
  }
}
```

---

### 2. 用户登录

**端点:** `POST /auth/login`

**请求参数:**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| email | String | 是 | 邮箱地址 |
| password | String | 是 | 密码 |

**响应示例:**

```json
{
  "success": true,
  "data": {
    "user_id": "user_123",
    "email": "user@example.com",
    "nickname": "张三",
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "dGhpcyBpcyBhIHJlZnJl...",
    "expires_at": "2026-03-07T14:30:00Z"
  }
}
```

---

### 3. 获取用户信息

**端点:** `GET /user/profile`

**响应示例:**

```json
{
  "success": true,
  "data": {
    "user_id": "user_123",
    "email": "user@example.com",
    "nickname": "张三",
    "avatar_url": "https://...",
    "subscription": {
      "plan": "premium",
      "status": "active",
      "expires_at": "2026-04-06"
    },
    "stats": {
      "total_sessions": 45,
      "total_minutes": 680,
      "average_score": 78.5
    }
  }
}
```

---

## 📊 数据导出 API

### 1. 导出个人数据

**端点:** `GET /user/export/data`

**描述:** 导出用户的所有 EEG 数据和分析结果（GDPR 合规）

**响应:** ZIP 文件下载

---

### 2. 删除个人数据

**端点:** `DELETE /user/data`

**描述:** 删除用户的所有数据（GDPR 被遗忘权）

**响应示例:**

```json
{
  "success": true,
  "message": "所有个人数据已删除"
}
```

---

## ❌ 错误码

| 错误码 | HTTP 状态码 | 描述 |
|--------|-----------|------|
| INVALID_REQUEST | 400 | 请求参数无效 |
| UNAUTHORIZED | 401 | 未认证或 Token 过期 |
| FORBIDDEN | 403 | 无权限访问 |
| NOT_FOUND | 404 | 资源不存在 |
| TASK_NOT_FOUND | 404 | 任务不存在 |
| SESSION_NOT_FOUND | 404 | 会话不存在 |
| INTERNAL_ERROR | 500 | 服务器内部错误 |
| EEG_PROCESSING_ERROR | 500 | EEG 处理失败 |
| DEVICE_ERROR | 500 | 设备连接失败 |

---

## 📝 使用示例

### Python 示例

```python
import requests
import websockets
import asyncio

# 1. 登录
response = requests.post('http://localhost:8000/api/v1/auth/login', json={
    'email': 'user@example.com',
    'password': 'password123'
})
token = response.json()['data']['token']

headers = {'Authorization': f'Bearer {token}'}

# 2. 上传 EEG 数据
with open('subject.edf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/eeg/upload',
        headers=headers,
        files={'file': f}
    )
    task_id = response.json()['data']['task_id']

# 3. 查询结果
import time
time.sleep(30)  # 等待处理完成

response = requests.get(
    f'http://localhost:8000/api/v1/eeg/analysis/{task_id}',
    headers=headers
)
result = response.json()['data']['result']
print(f"Attention: {result['attention_score']}")

# 4. 开始训练
response = requests.post(
    'http://localhost:8000/api/v1/training/sessions',
    headers=headers,
    json={'program_id': 'focus_basic'}
)
session_id = response.json()['data']['session_id']

# 5. WebSocket 实时训练
async def train():
    async with websockets.connect(
        f'ws://localhost:8000/api/v1/training/sessions/{session_id}'
    ) as ws:
        while True:
            # 发送 EEG 数据
            eeg_data = get_eeg_data()  # 从设备获取
            await ws.send(json.dumps({
                'type': 'eeg_data',
                'data': eeg_data.tolist()
            }))
            
            # 接收反馈
            response = await ws.recv()
            feedback = json.loads(response)
            print(f"Feedback: {feedback}")

asyncio.run(train())
```

---

**文档维护:** NeuroAI-Lab Team  
**联系方式:** support@neuroai-lab.com
