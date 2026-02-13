"""
RAG智能问诊助手 - 主服务
FastAPI + SSE流式输出
"""
import json
import asyncio
from typing import List, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import redis
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import settings
from models import ConsultRequest, ConsultResponse, KnowledgeSource, IncrementalUpdate, Document
from retriever import MultiPathRetriever
from mcp_tools import MCPToolManager
from knowledge_base import KnowledgeBase
from query_optimizer import optimize as optimize_query
from chat_history import get_messages, append_turn, messages_to_history_list


# 全局对象
retriever = None
mcp_manager = None
redis_client = None
knowledge_base = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global retriever, mcp_manager, redis_client, knowledge_base
    
    print("🚀 启动RAG智能问诊助手...")
    
    # 初始化组件
    retriever = MultiPathRetriever()
    mcp_manager = MCPToolManager()
    knowledge_base = KnowledgeBase()
    
    # 初始化Redis
    try:
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )
        redis_client.ping()
        print("✅ Redis连接成功")
    except Exception as e:
        print(f"⚠️  Redis连接失败: {e}")
        redis_client = None
    
    print("✅ 系统启动完成")
    
    yield
    
    # 清理资源
    print("👋 关闭系统...")
    if redis_client:
        redis_client.close()


app = FastAPI(
    title="RAG智能问诊助手",
    description="基于RAG的医疗问诊系统，支持多路召回、MCP工具兜底和SSE流式输出",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    print("⚠️  静态文件目录不存在，跳过挂载")



# LLM实例
llm = ChatOpenAI(
    openai_api_key=settings.openai_api_key,
    openai_api_base=settings.openai_api_base,
    model=settings.openai_model,
    temperature=0.7,
    streaming=True
)


def get_cache_key(user_id: str, question: str) -> str:
    """生成缓存key"""
    return f"consult:{user_id}:{hash(question)}"


def check_cache(cache_key: str) -> ConsultResponse:
    """检查缓存"""
    if not redis_client:
        return None
    
    try:
        cached = redis_client.get(cache_key)
        if cached:
            print("💾 命中缓存")
            data = json.loads(cached)
            return ConsultResponse(**data)
    except Exception as e:
        print(f"⚠️  缓存读取失败: {e}")
    
    return None


def set_cache(cache_key: str, response: ConsultResponse, ttl: int = 3600):
    """设置缓存"""
    if not redis_client:
        return
    
    try:
        redis_client.setex(
            cache_key,
            ttl,
            json.dumps(response.model_dump(), ensure_ascii=False)
        )
        print("💾 已缓存结果")
    except Exception as e:
        print(f"⚠️  缓存写入失败: {e}")


def get_request_history(request: ConsultRequest) -> List[dict]:
    """从 Redis 按 session_id 读取对话历史（不依赖前端传 history）"""
    if not request.session_id or not redis_client:
        return []
    raw = get_messages(request.session_id, redis_client)
    return messages_to_history_list(raw, max_turns=6)


def build_prompt(question: str, knowledge_sources: List[KnowledgeSource], history: List) -> str:
    """构建问诊提示词（history 为服务端从 Redis 拉取的最近几轮，格式 [{"role":"user"|"assistant","content":"..."}]）"""
    
    # 整理知识来源
    knowledge_text = ""
    for i, source in enumerate(knowledge_sources, 1):
        source_type = "【知识库】" if source.source == "knowledge_base" else "【联网搜索】"
        knowledge_text += f"\n{source_type} 来源{i}：\n{source.content}\n"
    
    # 历史对话上下文（若有）
    history_block = ""
    if history:
        lines = []
        for msg in history:
            role = (msg.get("role") or "").strip()
            content = (msg.get("content") or "").strip()
            if role and content:
                lines.append(f"{'用户' if role == 'user' else '助手'}：{content}")
        if lines:
            history_block = "历史对话：\n" + "\n".join(lines) + "\n\n"
    
    # 构建提示词
    system_prompt = """你是一个专业的医疗问诊助手，具备丰富的医学知识。你的任务是：

1. **理解病情**：仔细分析用户的症状描述
2. **信息补全**：如果信息不完整，主动询问关键信息（症状持续时间、严重程度、伴随症状等）
3. **知识检索**：基于提供的医疗知识，给出专业建议
4. **结构化建议**：提供清晰的分步建议

**回答要求**：
- 专业、准确、易懂
- 引用知识来源时标注【知识库】或【联网搜索】
- 给出3-5条结构化建议
- 必要时提醒用户就医

**重要提示**：
- 你不能替代专业医生诊断
- 紧急情况请立即就医
- 建议仅供参考"""

    user_prompt = f"""{history_block}用户问题：{question}

参考知识：
{knowledge_text}

请基于以上知识给出专业的问诊建议。"""
    
    return system_prompt, user_prompt


def extract_suggestions(answer: str) -> List[str]:
    """从回答中提取结构化建议"""
    suggestions = []
    
    # 尝试提取编号列表
    lines = answer.split('\n')
    for line in lines:
        line = line.strip()
        # 匹配 "1. xxx" 或 "- xxx" 格式
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            # 清理前缀
            suggestion = line.lstrip('0123456789.-•').strip()
            if suggestion:
                suggestions.append(suggestion)
    
    return suggestions[:5]  # 最多返回5条


async def stream_response(request: ConsultRequest) -> AsyncGenerator[str, None]:
    """SSE流式响应生成器"""
    
    try:
        # 0. 从 Redis 拉取对话历史（不依赖前端传 history）
        history = get_request_history(request)

        # 1. 提问优化（仅用于检索，回答与缓存仍用原问题）
        retrieval_query = optimize_query(
            request.question,
            history=history,
            enable_rewrite=settings.enable_query_rewrite,
            enable_normalize=settings.enable_query_normalize,
        )
        yield f"data: {json.dumps({'type': 'status', 'message': '正在检索医疗知识...'}, ensure_ascii=False)}\n\n"
        knowledge_sources = retriever.retrieve(retrieval_query)

        # 2. MCP工具兜底
        if not knowledge_sources or (knowledge_sources and max([s.score for s in knowledge_sources if s.score], default=0) < 0.5):
            yield f"data: {json.dumps({'type': 'status', 'message': '知识库信息不足，正在联网搜索...'}, ensure_ascii=False)}\n\n"
            knowledge_sources = await mcp_manager.enhance_retrieval(retrieval_query, knowledge_sources)
        
        # 3. 发送知识来源
        sources_data = [
            {
                'source': s.source,
                'content': s.content[:200] + '...' if len(s.content) > 200 else s.content,
                'score': s.score,
                'metadata': s.metadata
            }
            for s in knowledge_sources
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data}, ensure_ascii=False)}\n\n"
        
        # 4. 构建提示词（使用服务端历史）
        system_prompt, user_prompt = build_prompt(request.question, knowledge_sources, history)
        
        # 5. 流式生成回答
        yield f"data: {json.dumps({'type': 'status', 'message': '正在生成回答...'}, ensure_ascii=False)}\n\n"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        full_answer = ""
        async for chunk in llm.astream(messages):
            if chunk.content:
                full_answer += chunk.content
                yield f"data: {json.dumps({'type': 'content', 'content': chunk.content}, ensure_ascii=False)}\n\n"
        
        # 6. 提取建议
        suggestions = extract_suggestions(full_answer)
        
        # 7. 发送完成信号
        yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        
        # 8. 缓存结果（如果有user_id）
        if request.user_id:
            response = ConsultResponse(
                answer=full_answer,
                sources=knowledge_sources,
                suggestions=suggestions
            )
            cache_key = get_cache_key(request.user_id, request.question)
            set_cache(cache_key, response)

        # 9. 将本轮对话写入 Redis（若有 session_id）
        if request.session_id and redis_client:
            append_turn(request.session_id, request.question, full_answer, redis_client, ttl=settings.chat_history_ttl)
    
    except Exception as e:
        error_msg = f"生成回答时出错: {str(e)}"
        print(f"❌ {error_msg}")
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"


@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "RAG智能问诊助手",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/api/consult/stream")
async def consult_stream(request: ConsultRequest):
    """
    问诊接口（SSE流式）
    """
    # 检查缓存
    if request.user_id:
        cache_key = get_cache_key(request.user_id, request.question)
        cached_response = check_cache(cache_key)
        if cached_response:
            # 返回缓存的完整响应
            async def cached_stream():
                yield f"data: {json.dumps({'type': 'cached', 'message': '使用缓存结果'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'content', 'content': cached_response.answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': cached_response.suggestions}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            
            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
    
    # 流式响应
    return StreamingResponse(
        stream_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/consult", response_model=ConsultResponse)
async def consult(request: ConsultRequest):
    """
    问诊接口（非流式）
    """
    # 检查缓存
    if request.user_id:
        cache_key = get_cache_key(request.user_id, request.question)
        cached_response = check_cache(cache_key)
        if cached_response:
            return cached_response
    
    try:
        # 0. 从 Redis 拉取对话历史
        history = get_request_history(request)

        # 1. 提问优化（仅用于检索）
        retrieval_query = optimize_query(
            request.question,
            history=history,
            enable_rewrite=settings.enable_query_rewrite,
            enable_normalize=settings.enable_query_normalize,
        )
        knowledge_sources = retriever.retrieve(retrieval_query)

        # 2. MCP工具兜底
        knowledge_sources = await mcp_manager.enhance_retrieval(retrieval_query, knowledge_sources)

        # 3. 构建提示词（仍用原始问题，历史来自 Redis）
        system_prompt, user_prompt = build_prompt(request.question, knowledge_sources, history)
        
        # 4. 生成回答
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        answer = response.content
        
        # 5. 提取建议
        suggestions = extract_suggestions(answer)
        
        # 6. 构建响应
        result = ConsultResponse(
            answer=answer,
            sources=knowledge_sources,
            suggestions=suggestions
        )
        
        # 7. 缓存结果
        if request.user_id:
            set_cache(cache_key, result)

        # 8. 将本轮对话写入 Redis（若有 session_id）
        if request.session_id and redis_client:
            append_turn(request.session_id, request.question, answer, redis_client, ttl=settings.chat_history_ttl)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问诊失败: {str(e)}")


@app.post("/api/knowledge/build")
async def build_knowledge_base(file_path: str):
    """
    构建知识库（旧版 JSON 格式，如 data/medical_knowledge.json）
    """
    try:
        knowledge_base.build_knowledge_base(file_path)
        
        # 重建BM25索引
        retriever._build_bm25_index()
        
        return {"message": "知识库构建成功", "file": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"构建失败: {str(e)}")


@app.post("/api/knowledge/build_medical")
async def build_medical_knowledge(file_path: str = "data/medical.txt"):
    """
    从 medical.txt（JSONL 病症数据）构建病症库。
    会先删除同名 collection 再按新 schema 创建并写入数据。
    """
    try:
        knowledge_base.build_medical_knowledge_base(file_path)
        
        # 刷新检索器使用的 collection 并重建 BM25
        from pymilvus import Collection
        retriever.collection = Collection(settings.milvus_collection_name)
        retriever.collection.load()
        retriever._build_bm25_index()
        
        return {"message": "病症库构建成功", "file": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"构建失败: {str(e)}")


@app.post("/api/knowledge/update")
async def update_knowledge_base(request: IncrementalUpdate):
    """
    增量更新知识库
    """
    try:
        knowledge_base.incremental_update(request.documents, request.update_type)
        
        # 重建BM25索引
        retriever._build_bm25_index()
        
        return {
            "message": "增量更新成功",
            "update_type": request.update_type,
            "count": len(request.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
