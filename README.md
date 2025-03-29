# LLM请求处理系统

一个用于批量处理LLM请求、管理对话记录和模拟LLM响应的Python项目。

## 功能特性

- 支持多种LLM API（OpenAI、Claude等）
- 批量处理模型请求，支持多线程
- 自动错误处理和重试机制
- 对话记录和token使用统计存储
- 模拟LLM响应速度的FastAPI服务
- 支持流式响应和普通响应

## 安装和配置

1. 克隆项目：
```bash
git clone [项目地址]
cd simple_llm_request
```

2. 配置API密钥：
编辑`config.py`文件，设置以下变量：
- `api_base`: API基础URL
- `api_key`: API密钥
- `silicon_flow_api_base`: Silicon Flow API基础URL（可选）
- `silicon_flow_api_key`: Silicon Flow API密钥（可选）
- `ali_api_base`: 阿里API基础URL（可选）
- `ali_api_key`: 阿里API密钥（可选）
- `proxies`: 代理设置（可选）

## 使用示例

### 批量处理模型请求
```python
from make_a_request import batch_process_models

models = ["claude-3-5-sonnet", "gpt-4o", "deepseek-ai/DeepSeek-V3"]
prompt = "你的问题或提示"

results = batch_process_models(
    models=models,
    prompt=prompt,
    max_workers=5,
    stream=True
)
```

### 启动模拟服务器
```bash
uvicorn server:app --reload
```

## 项目结构

- `make_a_request.py`: 批量处理LLM请求的主程序
- `llm_client.py`: 与不同LLM API交互的客户端
- `database.py`: 对话记录和token使用统计的数据库管理
- `server.py`: 模拟LLM响应的FastAPI服务
- `config.py`: 配置文件（需自行创建）
- `*.db`: SQLite数据库文件

## 依赖项

- requests
- fastapi
- uvicorn
- sqlite3
- tiktoken
- sseclient

## 模型支持

### 思考型模型
- o3-mini
- gemini-2.0-flash-thinking-exp-01-21
- o1-mini
- qwq-32b
- qwq-plus-2025-03-05

### 常规模型
- claude-3-7-sonnet-20250219
- claude-3-5-sonnet
- gemini-2.0-flash
- deepseek-ai/DeepSeek-V3
- gpt-4.5-preview
- gpt-4o
- TA/Qwen/Qwen2.5-72B-Instruct-Turbo

## 数据库管理

对话记录和token使用统计存储在SQLite数据库中，可通过`database.py`中的`ConversationDB`类进行管理。

## 许可证

MIT License
