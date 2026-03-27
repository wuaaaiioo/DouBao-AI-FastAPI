import asyncio
import base64
import hashlib
import hmac
import json
import os
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
import pymysql
from pymysql.cursors import DictCursor
import tiktoken
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sse_starlette import EventSourceResponse


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env.shared")
load_dotenv(BASE_DIR / ".env")

def clean_env_string(value: str) -> str:
    return value.strip().strip('"').strip("'")


FRONTEND_ORIGIN = clean_env_string(os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Cache-Control"],
)

ZHIPU_API_KEY = "6444895464fc41739f775a5c385c0329.PU2D3vJmL7SeYcWu"
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_MODEL = "glm-4.7"
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "replace-with-a-strong-secret")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "xzl200318")
DB_NAME = os.getenv("DB_NAME", "AI_Chat")
enc = tiktoken.get_encoding("cl100k_base")


class TokenPayload(BaseModel):
    sub: str
    username: str | None = None
    name: str | None = None
    type: str | None = None
    iat: int | None = None
    exp: int | None = None


class MessagePayload(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("消息内容不能为空")
        return normalized


class SessionCreatePayload(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=100)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        normalized = value.strip()
        return normalized or "新会话"


class ChatPayload(BaseModel):
    messages: list[MessagePayload] = Field(min_length=1, max_length=200)
    session_id: str | None = Field(default=None, max_length=64)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def base64url_decode(value: str) -> bytes:
    normalized = value.replace("-", "+").replace("_", "/")
    padding = "=" * (-len(normalized) % 4)
    return base64.b64decode(normalized + padding)


def verify_access_token(authorization: str | None) -> TokenPayload:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少访问令牌")

    token = authorization.removeprefix("Bearer ").strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="访问令牌格式非法")

    header_part, payload_part, signature_part = parts
    signing_input = f"{header_part}.{payload_part}".encode("utf-8")
    expected_signature = hmac.new(
        TOKEN_SECRET.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    try:
        signature = base64url_decode(signature_part)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="访问令牌签名非法") from exc

    if len(signature) != len(expected_signature) or not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail="访问令牌签名非法")

    try:
        payload_data = json.loads(base64url_decode(payload_part).decode("utf-8"))
        payload = TokenPayload.model_validate(payload_data)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="访问令牌内容非法") from exc

    now = int(datetime.now(timezone.utc).timestamp())
    if payload.exp is not None and payload.exp < now:
        raise HTTPException(status_code=401, detail="访问令牌已过期")

    if payload.type and payload.type != "access":
        raise HTTPException(status_code=401, detail="访问令牌类型非法")

    return payload


def get_current_user(authorization: str | None = Header(default=None)) -> TokenPayload:
    return verify_access_token(authorization)


def get_conn():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
    )


def init_db() -> None:
    with closing(get_conn()) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SET SESSION innodb_lock_wait_timeout = 10")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话ID',
                    user_id BIGINT UNSIGNED NOT NULL COMMENT '所属用户ID',
                    title VARCHAR(120) NOT NULL DEFAULT '新对话' COMMENT '会话标题',
                    last_message_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最后一条消息时间',
                    is_deleted TINYINT NOT NULL DEFAULT 0 COMMENT '0未删除 1已删除',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY (id),
                    KEY idx_sessions_user_lastmsg (user_id, last_message_at DESC),
                    CONSTRAINT fk_sessions_user
                      FOREIGN KEY (user_id) REFERENCES users (id)
                      ON DELETE CASCADE ON UPDATE RESTRICT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息ID',
                    session_id BIGINT UNSIGNED NOT NULL COMMENT '会话ID',
                    role ENUM('system', 'user', 'assistant') NOT NULL COMMENT '消息角色',
                    content MEDIUMTEXT NOT NULL COMMENT '消息内容',
                    model_name VARCHAR(80) DEFAULT NULL COMMENT '模型名称',
                    prompt_tokens INT UNSIGNED DEFAULT NULL,
                    completion_tokens INT UNSIGNED DEFAULT NULL,
                    total_tokens INT UNSIGNED DEFAULT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id),
                    KEY idx_messages_session_created (session_id, created_at),
                    CONSTRAINT fk_messages_session
                      FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
                      ON DELETE CASCADE ON UPDATE RESTRICT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
        conn.commit()


@app.on_event("startup")
async def on_startup() -> None:
    init_db()


def truncate_messages(messages: list[MessagePayload], max_tokens: int = 60000):
    token_counts = []
    for msg in messages:
        token_count = len(enc.encode(msg.role)) + len(enc.encode(msg.content)) + 3
        token_counts.append(token_count)

    total_tokens = 0
    truncated_messages: list[MessagePayload] = []
    for i in reversed(range(len(messages))):
        if total_tokens + token_counts[i] > max_tokens:
            break
        total_tokens += token_counts[i]
        truncated_messages.append(messages[i])

    truncated_messages.reverse()
    if not truncated_messages and len(messages) > 0:
        truncated_messages = [messages[-1]]

    return truncated_messages


def create_session_record(user_id: str, title: str = "新会话") -> dict:
    safe_title = (title or "新会话").strip() or "新会话"

    with closing(get_conn()) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_sessions (user_id, title, last_message_at)
                VALUES (%s, %s, %s)
                """,
                (user_id, safe_title, utc_now()),
            )
            session_id = str(cursor.lastrowid)
        conn.commit()

    return {
        "id": session_id,
        "title": safe_title,
        "preview": "",
        "updatedAt": utc_now(),
    }


def get_session_or_404(conn, session_id: str, user_id: str):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM chat_sessions WHERE id = %s AND user_id = %s",
            (session_id, user_id),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="会话不存在")

    return row


def list_sessions(user_id: str, page: int, page_size: int) -> dict:
    offset = (page - 1) * page_size

    with closing(get_conn()) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS total FROM chat_sessions WHERE user_id = %s AND is_deleted = 0",
                (user_id,),
            )
            total = cursor.fetchone()["total"]
            cursor.execute(
                """
                SELECT
                    s.id,
                    s.title,
                    s.updated_at,
                    s.last_message_at,
                    COALESCE((
                        SELECT SUBSTRING(m.content, 1, 200)
                        FROM chat_messages m
                        WHERE m.session_id = s.id
                        ORDER BY m.created_at DESC, m.id DESC
                        LIMIT 1
                    ), '') AS preview
                FROM chat_sessions s
                WHERE s.user_id = %s AND s.is_deleted = 0
                ORDER BY s.last_message_at DESC, s.id DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, page_size, offset),
            )
            rows = cursor.fetchall()

    items = [
        {
            "id": str(row["id"]),
            "title": row["title"],
            "preview": row["preview"],
            "updatedAt": row["last_message_at"] or row["updated_at"],
        }
        for row in rows
    ]

    return {
        "items": items,
        "total": total,
        "hasMore": offset + len(items) < total,
    }


def encode_cursor(created_at: str, message_id: str) -> str:
    return f"{created_at}|{message_id}"


def decode_cursor(cursor: str | None) -> tuple[str, str] | None:
    if not cursor or "|" not in cursor:
        return None
    created_at, message_id = cursor.split("|", 1)
    return created_at, message_id


def list_messages(user_id: str, session_id: str, cursor: str | None, limit: int) -> dict:
    with closing(get_conn()) as conn:
        get_session_or_404(conn, session_id, user_id)
        params: list[str | int] = [session_id]
        query = """
            SELECT id, role, content, created_at
            FROM chat_messages
            WHERE session_id = %s
        """

        decoded_cursor = decode_cursor(cursor)
        if decoded_cursor:
            cursor_created_at, cursor_id = decoded_cursor
            query += " AND (created_at < %s OR (created_at = %s AND id < %s))"
            params.extend([cursor_created_at, cursor_created_at, cursor_id])

        query += " ORDER BY created_at DESC, id DESC LIMIT %s"
        params.append(limit + 1)
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    has_more = len(rows) > limit
    page_rows = rows[:limit]
    ordered_rows = list(reversed(page_rows))
    next_cursor = None

    if has_more and page_rows:
        last_row = page_rows[-1]
        next_cursor = encode_cursor(last_row["created_at"], last_row["id"])

    items = [
        {
            "id": str(row["id"]),
            "role": "assistant" if row["role"] == "assistant" else "user",
            "content": row["content"],
            "createdAt": row["created_at"],
        }
        for row in ordered_rows
    ]

    return {
        "items": items,
        "nextCursor": next_cursor,
        "hasMore": has_more,
    }


def maybe_update_session_title(conn, session_id: str, title_source: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT title FROM chat_sessions WHERE id = %s",
            (session_id,),
        )
        session_row = cursor.fetchone()

        if session_row and session_row["title"] in {"新会话", "新对话"}:
            next_title = (title_source or "新会话").strip()[:20] or "新会话"
            cursor.execute(
                "UPDATE chat_sessions SET title = %s WHERE id = %s",
                (next_title, session_id),
            )


def append_message(conn, session_id: str, role: str, content: str, created_at: str | None = None) -> str:
    now = created_at or utc_now()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, created_at)
            VALUES (%s, %s, %s, %s)
            """,
            (session_id, role, content, now),
        )
        return str(cursor.lastrowid)


def store_chat_turn(user_id: str, session_id: str | None, messages: list[MessagePayload], assistant_content: str) -> dict:
    latest_user_message = next(
        (msg for msg in reversed(messages) if msg.role == "user" and msg.content.strip()),
        None,
    )

    if not latest_user_message:
        raise HTTPException(status_code=400, detail="缺少用户消息")

    latest_user_content = latest_user_message.content.strip()
    now = utc_now()

    with closing(get_conn()) as conn:
        if session_id:
            get_session_or_404(conn, session_id, user_id)
        else:
            created = create_session_record(user_id, latest_user_content[:20] or "新会话")
            session_id = created["id"]

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            )
            last_message_row = cursor.fetchone()

        if not last_message_row or not (
            last_message_row["role"] == "user" and last_message_row["content"] == latest_user_content
        ):
            append_message(conn, session_id, "user", latest_user_content, now)

        append_message(conn, session_id, "assistant", assistant_content or "未获取到有效回答", utc_now())
        maybe_update_session_title(conn, session_id, latest_user_content)

        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE chat_sessions
                SET last_message_at = %s, updated_at = %s
                WHERE id = %s AND user_id = %s
                """,
                (utc_now(), utc_now(), session_id, user_id),
            )
        conn.commit()

    return {"session_id": session_id}


async def stream_zhipu_api(messages: list[MessagePayload]):
    truncated_msgs = truncate_messages(messages, max_tokens=60000)

    payload = {
        "model": ZHIPU_MODEL,
        "messages": [{"role": msg.role, "content": msg.content} for msg in truncated_msgs],
        "max_tokens": 65536,
        "temperature": 1.0,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", ZHIPU_API_URL, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    line_data = line.lstrip("data: ")
                    if line_data == "[DONE]":
                        break
                    try:
                        data = json.loads(line_data)
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                            await asyncio.sleep(0.01)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
    except httpx.TimeoutException:
        yield "请求上游模型超时，请稍后再试"
    except httpx.HTTPError as exc:
        message = str(exc)
        if "429" in message:
            yield "调用频率过高，请稍后再试"
        elif "context length exceeded" in message.lower():
            yield "上下文过长，自动保留最近消息继续对话"
        else:
            yield f"调用智谱失败：{message}"


async def call_zhipu_api(messages: list[MessagePayload]):
    truncated_msgs = truncate_messages(messages, max_tokens=60000)

    payload = {
        "model": ZHIPU_MODEL,
        "messages": [{"role": msg.role, "content": msg.content} for msg in truncated_msgs],
        "max_tokens": 65536,
        "temperature": 1.0,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(ZHIPU_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="调用智谱超时") from exc
    except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as exc:
        message = str(exc)
        if "context length exceeded" in message.lower():
            raise HTTPException(status_code=500, detail="上下文过长，自动保留最近消息继续对话") from exc
        raise HTTPException(status_code=500, detail=f"调用智谱失败：{message}") from exc


@app.get("/chat/sessions")
async def get_chat_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=50),
    current_user: TokenPayload = Depends(get_current_user),
):
    return {
        "code": 200,
        "msg": "成功",
        "data": list_sessions(current_user.sub, page, page_size),
    }


@app.post("/chat/sessions")
async def create_chat_session(
    payload: SessionCreatePayload,
    current_user: TokenPayload = Depends(get_current_user),
):
    created = create_session_record(current_user.sub, payload.title)
    return {
        "code": 200,
        "msg": "成功",
        "data": created,
    }


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: TokenPayload = Depends(get_current_user),
):
    with closing(get_conn()) as conn:
        get_session_or_404(conn, session_id, current_user.sub)
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, current_user.sub))
        conn.commit()

    return {
        "code": 200,
        "msg": "成功",
        "data": True,
    }


@app.get("/chat/sessions/{session_id}/messages")
async def get_chat_session_messages(
    session_id: str,
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: TokenPayload = Depends(get_current_user),
):
    return {
        "code": 200,
        "msg": "成功",
        "data": list_messages(current_user.sub, session_id, cursor, limit),
    }


@app.post("/chat/stream")
async def chat_stream(
    payload: ChatPayload = Body(...),
    current_user: TokenPayload = Depends(get_current_user),
):
    async def event_generator():
        full_content = ""
        stored_session_id: str | None = None
        try:
            async for chunk in stream_zhipu_api(payload.messages):
                full_content += chunk
                yield {"data": chunk}
        except Exception as exc:
            yield {"data": f"流式响应异常：{str(exc)}"}
        finally:
            try:
                stored = store_chat_turn(current_user.sub, payload.session_id, payload.messages, full_content)
                stored_session_id = str(stored["session_id"])
            except Exception as exc:
                print(f"store chat turn failed: {exc}")
        if stored_session_id:
            yield {"data": json.dumps({"type": "session_created", "session_id": stored_session_id}, ensure_ascii=False)}
        yield {"data": "[DONE]"}

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
    )


@app.post("/chat")
async def chat(
    payload: ChatPayload = Body(...),
    current_user: TokenPayload = Depends(get_current_user),
):
    ai_reply = await call_zhipu_api(payload.messages)
    stored = store_chat_turn(current_user.sub, payload.session_id, payload.messages, ai_reply)
    return {
        "success": True,
        "data": {
            "reply": ai_reply,
            "session_id": stored["session_id"],
        },
        "message": "成功",
    }


@app.get("/")
async def root():
    return {"message": "FastAPI 聊天与会话持久化接口已启动"}
