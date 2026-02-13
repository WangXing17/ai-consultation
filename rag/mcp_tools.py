"""
MCP工具模块
提供Bing搜索兜底能力，当知识库无法回答时触发联网检索
"""
import httpx
from typing import List, Optional
from config import settings
from models import KnowledgeSource


class BingSearchTool:
    """Bing搜索工具（MCP兜底）"""
    
    def __init__(self):
        self.api_key = settings.bing_search_api_key
        self.endpoint = settings.bing_search_endpoint
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
    
    def should_trigger(self, knowledge_sources: List[KnowledgeSource], confidence_score: float = 0.5) -> bool:
        """
        判断是否需要触发Bing搜索
        触发条件：
        1. 知识库未命中（没有检索结果）
        2. 检索结果相似度分数过低
        """
        # 条件1：没有检索结果
        if not knowledge_sources:
            print("🔍 知识库未命中，触发Bing搜索")
            return True
        
        # 条件2：所有结果分数都低于阈值
        max_score = max([s.score for s in knowledge_sources if s.score], default=0)
        if max_score < confidence_score:
            print(f"🔍 知识库分数过低({max_score:.2f} < {confidence_score})，触发Bing搜索")
            return True
        
        return False
    
    async def search(self, query: str, count: int = 3) -> List[KnowledgeSource]:
        """
        执行Bing搜索
        返回搜索结果作为知识来源
        """
        if not self.api_key or self.api_key == "your_bing_search_key":
            print("⚠️  Bing API Key未配置，跳过搜索")
            return []
        
        try:
            params = {
                "q": f"{query} 医疗健康",  # 添加医疗领域限定
                "count": count,
                "mkt": "zh-CN",
                "responseFilter": "Webpages"
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.endpoint,
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
            
            # 解析搜索结果
            sources = []
            if "webPages" in data and "value" in data["webPages"]:
                for item in data["webPages"]["value"][:count]:
                    source = KnowledgeSource(
                        source="bing_search",
                        content=item.get("snippet", ""),
                        score=None,  # Bing搜索不提供相似度分数
                        metadata={
                            "title": item.get("name", ""),
                            "url": item.get("url", ""),
                            "retrieval_type": "web_search"
                        }
                    )
                    sources.append(source)
            
            print(f"🌐 Bing搜索返回 {len(sources)} 条结果")
            return sources
        
        except httpx.HTTPStatusError as e:
            print(f"❌ Bing搜索HTTP错误: {e.response.status_code}")
            return []
        except Exception as e:
            print(f"❌ Bing搜索失败: {e}")
            return []
    
    def format_search_results(self, sources: List[KnowledgeSource]) -> str:
        """格式化搜索结果用于LLM"""
        if not sources:
            return ""
        
        formatted = "【联网搜索结果】\n"
        for i, source in enumerate(sources, 1):
            title = source.metadata.get("title", "")
            url = source.metadata.get("url", "")
            content = source.content
            formatted += f"\n{i}. {title}\n{content}\n来源: {url}\n"
        
        return formatted


class MCPToolManager:
    """MCP工具管理器"""
    
    def __init__(self):
        self.bing_tool = BingSearchTool()
    
    async def enhance_retrieval(
        self, 
        query: str, 
        knowledge_sources: List[KnowledgeSource]
    ) -> List[KnowledgeSource]:
        """
        增强检索结果
        当知识库不足时，使用Bing搜索兜底
        """
        # 判断是否需要触发搜索
        if not self.bing_tool.should_trigger(knowledge_sources):
            return knowledge_sources
        
        # 执行Bing搜索
        search_results = await self.bing_tool.search(query)
        
        # 合并结果
        enhanced_sources = knowledge_sources + search_results
        
        return enhanced_sources
