"""
用户提问优化模块
提供 Query Rewriting（LLM 改写）与关键词规范化，提升 RAG 检索效果。
详见 docs/query_optimization.md
"""
from typing import List, Optional
from langchain_openai import ChatOpenAI
from config import settings


# 医疗常见同义词/口语 → 规范表述（便于 BM25/规则 命中）
MEDICAL_SYNONYM_MAP = {
    "头疼": "头痛",
    "脑袋疼": "头痛",
    "脑袋痛": "头痛",
    "发烧": "发热",
    "高烧": "发热",
    "低烧": "低热",
    "咳嗽": "咳嗽",
    "肚子疼": "腹痛",
    "肚子痛": "腹痛",
    "胃疼": "胃痛",
    "拉肚子": "腹泻",
    "拉稀": "腹泻",
    "恶心想吐": "恶心 呕吐",
    "想吐": "恶心",
    "浑身没劲": "乏力",
    "没力气": "乏力",
    "感冒": "感冒",
    "流感": "流行性感冒",
    "消炎药": "抗生素",
    "退烧药": "解热镇痛药",
    "止痛药": "镇痛药",
    "降压药": "抗高血压药",
}


def normalize_keywords(query: str) -> str:
    """
    关键词规范化：将口语/同义词替换为知识库中更常见的表述，提升 BM25/规则召回。
    """
    if not query or not query.strip():
        return query
    text = query.strip()
    for colloquial, standard in MEDICAL_SYNONYM_MAP.items():
        if colloquial in text:
            text = text.replace(colloquial, standard)
    return text


def rewrite_query_for_retrieval(
    question: str,
    history: Optional[List] = None,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """
    使用 LLM 将用户问题改写成更利于检索的表述（保留医学关键信息、补全指代）。
    若 LLM 调用失败或未配置，则返回原问题。
    """
    if not question or not question.strip():
        return question

    if llm is None:
        llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            model=settings.openai_model,
            temperature=0.1,
        )

    history_context = ""
    if history and len(history) > 0:
        recent = history[-6:]  # 最近几轮
        parts = []
        for msg in recent:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            if role and content:
                parts.append(f"{role}: {content}")
        if parts:
            history_context = "最近对话：\n" + "\n".join(parts) + "\n\n"

    prompt = f"""你是一个医疗问诊检索助手。请将用户的提问改写成一句「仅包含医学相关关键信息的检索用问句」，用于在医疗知识库中检索。

要求：
1. 保留症状、部位、药物、疾病、检查等关键信息；
2. 若有指代（如「这个药」「上面的症状」），结合上下文替换为具体内容；
3. 去掉礼貌用语、语气词，输出简短一句，不要解释；
4. 若问题已清晰且无指代，可稍作同义替换（如「头疼」→「头痛」）以利检索；
5. 只输出改写后的一句话，不要加引号或前缀。

{history_context}用户当前问题：{question.strip()}

改写后的检索用问句："""

    try:
        response = llm.invoke(prompt)
        rewritten = (response.content or "").strip()
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"⚠️ Query 改写失败，使用原问题: {e}")
    return question


def optimize(
    question: str,
    history: Optional[List] = None,
    *,
    enable_rewrite: bool = True,
    enable_normalize: bool = True,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """
    统一入口：先改写（可选），再规范化（可选），返回用于检索的 query。
    回答与缓存仍应使用原始 question。
    """
    if not question or not question.strip():
        return question

    q = question.strip()
    if enable_rewrite:
        q = rewrite_query_for_retrieval(q, history=history, llm=llm)
    if enable_normalize:
        q = normalize_keywords(q)
    return q
