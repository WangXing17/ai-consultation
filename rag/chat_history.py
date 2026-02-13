"""
对话历史存储：使用 Redis 存储上下文，不依赖前端传 history。
与 LangChain 的 ChatMessageHistory 语义一致（role + content），便于与 query_optimizer、prompt 等配合。
调用方需传入 redis_client，避免循环依赖。
"""
import json
from typing import List, Any

# Redis key 前缀、默认 TTL（秒）
CHAT_HISTORY_KEY_PREFIX = "chat_history:"
DEFAULT_TTL = 86400  # 24 小时


def _key(session_id: str) -> str:
    return f"{CHAT_HISTORY_KEY_PREFIX}{session_id}"


def get_messages(session_id: str, redis_client: Any) -> List[dict]:
    """
    从 Redis 读取该会话的历史消息。
    返回 [{"role": "user"|"assistant", "content": "..."}, ...]，按时间顺序。
    """
    if not redis_client:
        return []
    try:
        raw = redis_client.get(_key(session_id))
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠️ 读取对话历史失败: {e}")
        return []


def add_user_message(session_id: str, content: str, redis_client: Any, ttl: int = DEFAULT_TTL) -> None:
    """追加一条用户消息并刷新 TTL"""
    if not redis_client:
        return
    key = _key(session_id)
    try:
        messages = get_messages(session_id, redis_client)
        messages.append({"role": "user", "content": content})
        redis_client.setex(key, ttl, json.dumps(messages, ensure_ascii=False))
    except Exception as e:
        print(f"⚠️ 写入用户消息失败: {e}")


def add_ai_message(session_id: str, content: str, redis_client: Any, ttl: int = DEFAULT_TTL) -> None:
    """追加一条助手消息并刷新 TTL"""
    if not redis_client:
        return
    key = _key(session_id)
    try:
        messages = get_messages(session_id, redis_client)
        messages.append({"role": "assistant", "content": content})
        redis_client.setex(key, ttl, json.dumps(messages, ensure_ascii=False))
    except Exception as e:
        print(f"⚠️ 写入助手消息失败: {e}")


def append_turn(session_id: str, user_content: str, ai_content: str, redis_client: Any, ttl: int = DEFAULT_TTL) -> None:
    """一次性追加一轮对话（用户问 + 助手答），减少两次 Redis 读写"""
    if not redis_client:
        return
    key = _key(session_id)
    try:
        messages = get_messages(session_id, redis_client)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": ai_content})
        redis_client.setex(key, ttl, json.dumps(messages, ensure_ascii=False))
    except Exception as e:
        print(f"⚠️ 写入对话轮次失败: {e}")


def clear_history(session_id: str, redis_client: Any) -> None:
    """清空该会话历史"""
    if not redis_client:
        return
    try:
        redis_client.delete(_key(session_id))
    except Exception as e:
        print(f"⚠️ 清空对话历史失败: {e}")


def messages_to_history_list(messages: List[dict], max_turns: int = 6) -> List[dict]:
    """
    将 Redis 中的 messages 转为 query_optimizer / build_prompt 使用的「最近 N 条」列表。
    每条为 {"role": "user"|"assistant", "content": "..."}。
    """
    if not messages:
        return []
    # 取最近 max_turns 条（按条数，不是按轮数）
    recent = messages[-max_turns:] if len(messages) > max_turns else messages
    return recent
