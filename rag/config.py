"""
配置管理模块
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """应用配置"""
    
    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Milvus配置
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "medical_knowledge")
    
    # Redis配置
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    
    # Bing搜索配置
    bing_search_api_key: str = os.getenv("BING_SEARCH_API_KEY", "")
    bing_search_endpoint: str = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
    
    # 检索配置
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "3"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    # 提问优化（Query Rewriting + 关键词规范化，仅用于检索，回答与缓存仍用原问题）
    enable_query_rewrite: bool = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() in ("1", "true", "yes")
    enable_query_normalize: bool = os.getenv("ENABLE_QUERY_NORMALIZE", "true").lower() in ("1", "true", "yes")

    # 对话历史（Redis 存储，按 session_id 读写）
    chat_history_ttl: int = int(os.getenv("CHAT_HISTORY_TTL", "86400"))  # 秒，默认 24 小时
    
    # 向量维度
    embedding_dim: int = 1536  # text-embedding-ada-002的维度
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
