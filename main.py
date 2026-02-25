from fastapi import FastAPI, HTTPException, Body  # 新增 Body 导入
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import asyncio  # 新增：异步支持
import tiktoken  # 用来计算 Token 数（智谱兼容 OpenAI 的 Token 计算规则）
from sse_starlette import EventSourceResponse  # 新增：支持SSE流式响应

# 初始化 FastAPI 应用
app = FastAPI()

# 配置跨域（必须！否则前端报跨域错误）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许所有来源，生产环境替换为你的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- 智谱 GLM-5 配置 --------------------------
# 替换成你的智谱 API Key
ZHIPU_API_KEY = "6444895464fc41739f775a5c385c0329.PU2D3vJmL7SeYcWu"
# 智谱 GLM-5 API 地址（和 curl 里的一致）
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
# 智谱模型名（glm-4.7 保持不变）
ZHIPU_MODEL = "glm-4.7"

# -------------------------- Token 截断逻辑（保留原有） --------------------------
# 初始化 Token 计算器（用 gpt-3.5-turbo 模型的规则，和智谱兼容）
enc = tiktoken.get_encoding("cl100k_base")

def truncate_messages(messages: list, max_tokens: int = 60000):
    """
    截断历史消息，确保总 Token 不超过 max_tokens（预留 5536 Token 给输出）
    :param messages: 历史消息列表 [{"role":"user/ai","content":"内容"}]
    :param max_tokens: 输入最大 Token 数（建议设为 60000，留 5536 给输出）
    :return: 截断后的消息列表
    """
    # 1. 计算每条消息的 Token 数
    token_counts = []
    for msg in messages:
        # 每条消息的 Token = role 长度 + content 长度 + 固定开销（3 个 Token）
        token_count = len(enc.encode(msg["role"])) + len(enc.encode(msg["content"])) + 3
        token_counts.append(token_count)
    
    # 2. 从后往前累加，保留最近的消息，直到接近上限
    total_tokens = 0
    truncated_messages = []
    # 倒序遍历（从最新的消息开始）
    for i in reversed(range(len(messages))):
        if total_tokens + token_counts[i] > max_tokens:
            break  # 超过上限，停止添加
        total_tokens += token_counts[i]
        truncated_messages.append(messages[i])
    
    # 3. 反转回来（恢复正序）
    truncated_messages.reverse()
    
    # 4. 如果全部截断后还是超，只保留最后 1 轮对话
    if not truncated_messages and len(messages) > 0:
        truncated_messages = [messages[-1]]
    
    return truncated_messages

# -------------------------- 核心改造：流式调用智谱 API --------------------------
async def stream_zhipu_api(messages: list):
    """异步流式调用智谱API，修复编码问题，逐段返回内容"""
    # 1. 先截断消息（防超限，保留原有逻辑）
    truncated_msgs = truncate_messages(messages, max_tokens=60000)
    
    # 2. 构造官方格式的 payload（开启stream=true）
    payload = {
        "model": ZHIPU_MODEL,  # 保持你的模型名
        "messages": truncated_msgs,
        "max_tokens": 65536,
        "temperature": 1.0,
        "stream": True  # 开启流式输出（关键）
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        # 发送流式请求，开启stream=True
        with requests.post(
            ZHIPU_API_URL,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            stream=True,  # 开启流式响应
            timeout=60
        ) as response:
            response.raise_for_status()
            # 逐行解析流式数据
            for line in response.iter_lines():
                if not line:
                    continue
                # 核心修复：强制用UTF-8解码，解决latin-1编码错误
                line_data = line.decode("utf-8").lstrip("data: ")
                if line_data == "[DONE]":  # 流式结束标记
                    break
                try:
                    data = json.loads(line_data)
                    # 提取分段的回复内容
                    content = data["choices"][0]["delta"].get("content", "")
                    if content:
                        # 确保内容是UTF-8编码的字符串，避免编码传递错误
                        yield content.encode("utf-8").decode("utf-8")
                        await asyncio.sleep(0.01)  # 可选：控制打字速度（数值越大越慢）
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.RequestException as e:
        # 错误信息也强制UTF-8编码，避免中文报错乱码
        error_msg = ""
        if "429" in str(e):
            error_msg = "调用频率过高，请稍后再试"
        elif "context length exceeded" in str(e).lower():
            error_msg = "上下文过长，自动保留最近消息继续对话"
        else:
            error_msg = f"调用智谱失败：{str(e)}"
        yield error_msg.encode("utf-8").decode("utf-8")

# -------------------------- 保留原有非流式接口（备用） --------------------------
def call_zhipu_api(messages: list):
    """原有非流式调用（备用）"""
    truncated_msgs = truncate_messages(messages, max_tokens=60000)
    
    payload = {
        "model": ZHIPU_MODEL,
        "messages": truncated_msgs,
        "max_tokens": 65536,
        "temperature": 1.0,
        "stream": False  # 关闭流式
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            ZHIPU_API_URL,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        ai_reply = result["choices"][0]["message"]["content"]
        return ai_reply

    except requests.exceptions.RequestException as e:
        if "context length exceeded" in str(e).lower():
            raise HTTPException(status_code=500, detail="上下文过长，自动保留最近消息继续对话")
        raise HTTPException(status_code=500, detail=f"调用智谱失败：{str(e)}")

# -------------------------- 接口改造：新增流式接口 + 保留原有接口 --------------------------
# 新增：流式接口（前端调用这个实现打字机效果）
@app.post("/chat/stream")
async def chat_stream(
    messages: list = Body(...),
    user_id: str = Body(default="default_user")
):
    # 返回SSE流式响应，指定UTF-8编码，彻底解决中文编码问题
    return EventSourceResponse(
        stream_zhipu_api(messages),
        media_type="text/event-stream; charset=utf-8"
    )

# 保留：原有非流式接口（备用，不影响旧逻辑）
@app.post("/chat")  
async def chat(
    messages: list = Body(...),
    user_id: str = Body(default="default_user")
):
    """接收历史消息列表，实现多轮对话（非流式）"""
    try:
        ai_reply = call_zhipu_api(messages)
        return {
            "success": True,
            "data": {"reply": ai_reply},
            "message": "成功"
        }
    except HTTPException as e:
        return {
            "success": False,
            "data": None,
            "message": e.detail
        }

@app.get("/")
async def root():
    return {"message": "智谱 GLM-4 流式+非流式多轮对话对接成功！"}

# 启动命令不变：uvicorn main:app --host 0.0.0.0 --port 8000 --reload