"""
数据模型定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    """对话消息"""
    role: str = Field(..., description="角色：user/assistant/system")
    content: str = Field(..., description="消息内容")


class ConsultRequest(BaseModel):
    """问诊请求"""
    question: str = Field(..., description="用户病情描述或问题")
    history: Optional[List[Message]] = Field(default=[], description="历史对话记录")
    user_id: Optional[str] = Field(default=None, description="用户ID，用于缓存")


class KnowledgeSource(BaseModel):
    """知识来源"""
    source: str = Field(..., description="来源：knowledge_base/bing_search")
    content: str = Field(..., description="知识内容")
    score: Optional[float] = Field(default=None, description="相似度分数")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="元数据")


class ConsultResponse(BaseModel):
    """问诊响应"""
    answer: str = Field(..., description="回答内容")
    sources: List[KnowledgeSource] = Field(default=[], description="知识来源")
    suggestions: List[str] = Field(default=[], description="结构化建议")


class Document(BaseModel):
    """文档模型"""
    id: Optional[str] = None
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    embedding: Optional[List[float]] = None


class IncrementalUpdate(BaseModel):
    """增量更新请求"""
    documents: List[Document] = Field(..., description="待更新的文档列表")
    update_type: str = Field(..., description="更新类型：add/update/delete")
