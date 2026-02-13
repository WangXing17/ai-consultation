"""
检索器模块
实现多路召回（语义向量检索 + 关键词/倒排检索 + 规则召回）与重排策略
"""
import jieba
import re
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from pymilvus import Collection, connections
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import settings
from models import KnowledgeSource


def _jieba_tokenize_one(text: str) -> List[str]:
    """单条文本分词，供多进程 Pool.map 使用（须为模块级函数以支持 pickle）"""
    return list(jieba.cut(text or ""))


class MultiPathRetriever:
    """多路召回检索器"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            model=settings.embedding_model
        )
        
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            model=settings.openai_model,
            temperature=0.1
        )
        
        # 连接Milvus
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            self.collection = Collection(settings.milvus_collection_name)
            self.collection.load()
            print(f"✅ Milvus连接成功，collection: {settings.milvus_collection_name}")
        except Exception as e:
            print(f"⚠️  Milvus连接失败: {e}")
            if "not exist" in str(e):
                print(f"💡 提示：知识库尚未构建，请先运行：")
                print(f"   python build_knowledge.py")
            self.collection = None
        
        # BM25索引（用于关键词检索）
        self.bm25_index = None
        self.bm25_docs = []
        self._build_bm25_index()
        
        # 医疗关键词规则库
        self.medical_rules = self._load_medical_rules()
    
    # 病症库 schema 的字段（medical.txt 结构）
    MEDICAL_OUTPUT_FIELDS = ["id", "content", "name", "category_primary", "symptoms", "cure_department", "cure_way", "get_way", "cured_prob"]
    
    # 从 Milvus 分批拉取时的每批条数
    MILVUS_QUERY_BATCH_SIZE = 2000
    
    def _build_bm25_index(self):
        """构建BM25索引用于关键词检索（使用病症库 schema 字段，分批从 Milvus 拉取）"""
        if not self.collection:
            return
        
        try:
            results = []
            # 优先使用 query_iterator 分批拉取，避免单次 query 数据量过大
            if hasattr(self.collection, "query_iterator"):
                it = self.collection.query_iterator(
                    batch_size=self.MILVUS_QUERY_BATCH_SIZE,
                    limit=-1,
                    expr="id != ''",
                    output_fields=self.MEDICAL_OUTPUT_FIELDS,
                )
                while True:
                    batch = it.next()
                    if not batch:
                        it.close()
                        break
                    results.extend(batch)
                    if len(batch) < self.MILVUS_QUERY_BATCH_SIZE:
                        break
            else:
                # 兼容无 query_iterator 时：分批 query，用 id not in 排除已取
                fetched_ids = set()
                while True:
                    if fetched_ids:
                        exclude = ", ".join(f'"{x}"' for x in fetched_ids)
                        expr = f"id not in [{exclude}]"
                    else:
                        expr = "id != ''"
                    batch = self.collection.query(
                        expr=expr,
                        output_fields=self.MEDICAL_OUTPUT_FIELDS,
                        limit=self.MILVUS_QUERY_BATCH_SIZE,
                    )
                    if not batch:
                        break
                    for doc in batch:
                        fid = doc.get("id")
                        if fid and fid not in fetched_ids:
                            fetched_ids.add(fid)
                            results.append(doc)
                    if len(batch) < self.MILVUS_QUERY_BATCH_SIZE:
                        break
                    if len(results) >= 50000:  # 安全上限
                        break
            
            self.bm25_docs = results

            # 分词：多进程并行加速，文档少时直接用主进程避免进程开销
            contents = [doc.get("content") or "" for doc in results]
            n_docs = len(contents)
            n_workers = min(max(1, cpu_count() - 1), n_docs, 8)
            if n_workers <= 1 or n_docs < 100:
                tokenized_docs = [_jieba_tokenize_one(t) for t in contents]
            else:
                with Pool(n_workers) as pool:
                    tokenized_docs = pool.map(_jieba_tokenize_one, contents, chunksize=max(1, n_docs // (n_workers * 4)))

            # 构建BM25索引
            if tokenized_docs:
                self.bm25_index = BM25Okapi(tokenized_docs)
                print(f"✅ BM25索引构建完成，包含 {len(tokenized_docs)} 个文档")
        except Exception as e:
            print(f"⚠️  BM25索引构建失败: {e}")
    
    def _load_medical_rules(self) -> Dict[str, List[str]]:
        """
        加载医疗规则库
        根据关键词触发特定的知识检索
        """
        return {
            "症状": ["发烧", "咳嗽", "头痛", "腹痛", "恶心", "呕吐", "腹泻", "乏力"],
            "疾病": ["感冒", "流感", "肺炎", "胃炎", "高血压", "糖尿病", "冠心病"],
            "药物": ["阿司匹林", "布洛芬", "对乙酰氨基酚", "抗生素", "降压药"],
            "检查": ["血常规", "尿常规", "CT", "核磁共振", "B超", "X光"],
            "紧急": ["急救", "中毒", "骨折", "出血", "休克", "昏迷"]
        }
    
    def vector_search(self, query: str, top_k: int = 10) -> List[KnowledgeSource]:
        """
        路径1：语义向量检索
        使用embedding进行相似度搜索
        """
        if not self.collection:
            return []
        
        try:
            # 向量化查询
            query_embedding = self.embeddings.embed_query(query)
            
            # 向量搜索（病症库 schema）
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=self.MEDICAL_OUTPUT_FIELDS
            )
            
            # 转换结果
            sources = []
            for hit in results[0]:
                # Milvus L2距离，越小越相似，转换为相似度分数
                similarity = 1 / (1 + hit.distance)
                
                if similarity >= settings.similarity_threshold:
                    entity = hit.entity
                    content = entity.get("content") or ""
                    name = entity.get("name") or ""
                    # 展示时带上疾病名称
                    display = f"【{name}】\n{content}" if name else content
                    source = KnowledgeSource(
                        source="knowledge_base",
                        content=display,
                        score=float(similarity),
                        metadata={
                            "retrieval_type": "vector",
                            "name": name,
                            "category_primary": entity.get("category_primary"),
                            "symptoms": entity.get("symptoms"),
                            "cure_department": entity.get("cure_department"),
                            "cure_way": entity.get("cure_way"),
                            "get_way": entity.get("get_way"),
                            "cured_prob": entity.get("cured_prob"),
                        }
                    )
                    sources.append(source)
            
            print(f"📊 向量检索返回 {len(sources)} 条结果")
            return sources
        except Exception as e:
            print(f"❌ 向量检索失败: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[KnowledgeSource]:
        """
        路径2：关键词/倒排检索（BM25）
        基于词频和逆文档频率的检索
        """
        if not self.bm25_index or not self.bm25_docs:
            return []
        
        try:
            # 分词
            query_tokens = list(jieba.cut(query))
            
            # BM25检索
            scores = self.bm25_index.get_scores(query_tokens)
            
            # 获取top-k结果
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            sources = []
            for idx in top_indices:
                score = scores[idx]
                if score > 0:  # BM25分数大于0
                    doc = self.bm25_docs[idx]
                    content = doc.get("content") or ""
                    name = doc.get("name") or ""
                    display = f"【{name}】\n{content}" if name else content
                    source = KnowledgeSource(
                        source="knowledge_base",
                        content=display,
                        score=float(score),
                        metadata={
                            "retrieval_type": "keyword",
                            "name": name,
                            "category_primary": doc.get("category_primary"),
                            "symptoms": doc.get("symptoms"),
                            "cure_department": doc.get("cure_department"),
                            "cure_way": doc.get("cure_way"),
                            "get_way": doc.get("get_way"),
                            "cured_prob": doc.get("cured_prob"),
                        }
                    )
                    sources.append(source)
            
            print(f"📊 关键词检索返回 {len(sources)} 条结果")
            return sources
        except Exception as e:
            print(f"❌ 关键词检索失败: {e}")
            return []
    
    def rule_based_search(self, query: str) -> Tuple[List[KnowledgeSource], str]:
        """
        路径3：规则召回
        基于医疗关键词规则触发特定检索
        """
        matched_category = None
        matched_keywords = []
        
        # 检查是否匹配规则
        for category, keywords in self.medical_rules.items():
            for keyword in keywords:
                if keyword in query:
                    matched_category = category
                    matched_keywords.append(keyword)
        
        if not matched_category:
            return [], None
        
        # 如果匹配到紧急情况，优先返回
        if matched_category == "紧急":
            print(f"⚠️  检测到紧急情况关键词: {matched_keywords}")
        
        # 病症库 schema 无「症状/疾病/药物」等 category 字段，规则仅作关键词匹配，不单独查库
        # 向量检索和关键词检索已会命中相关内容，这里直接返回空，避免按旧 schema 查库报错
        return [], matched_category
    
    def rerank(self, query: str, sources: List[KnowledgeSource], top_k: int = 3) -> List[KnowledgeSource]:
        """
        重排策略
        使用LLM对检索结果进行重排，选择最相关的top-k
        """
        if len(sources) <= top_k:
            return sources
        
        try:
            # 构建重排提示
            candidates = "\n\n".join([
                f"[{i}] {source.content[:200]}..." 
                for i, source in enumerate(sources)
            ])
            
            prompt = f"""你是一个医疗问诊助手。用户问题是：{query}

以下是候选知识片段：
{candidates}

请根据相关性对这些知识片段排序，返回最相关的{top_k}个片段的序号，用逗号分隔。
只返回序号，不要其他内容。例如：0,3,5"""
            
            response = self.llm.invoke(prompt)
            indices_str = response.content.strip()
            
            # 解析序号
            indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
            indices = [idx for idx in indices if 0 <= idx < len(sources)][:top_k]
            
            # 重排后的结果
            reranked = [sources[idx] for idx in indices]
            
            print(f"📊 重排后返回 {len(reranked)} 条结果")
            return reranked
        except Exception as e:
            print(f"⚠️  重排失败，返回原始结果: {e}")
            # 降级策略：按分数排序
            sorted_sources = sorted(sources, key=lambda x: x.score or 0, reverse=True)
            return sorted_sources[:top_k]
    
    def retrieve(self, query: str, top_k: int = None) -> List[KnowledgeSource]:
        """
        多路召回主函数
        整合向量检索、关键词检索和规则检索的结果
        """
        if top_k is None:
            top_k = settings.top_k_rerank
        
        print(f"🔍 开始多路召回检索，query: {query}")
        
        all_sources = []
        
        # 路径1：向量检索
        vector_results = self.vector_search(query, top_k=settings.top_k_retrieval)
        all_sources.extend(vector_results)
        
        # 路径2：关键词检索
        keyword_results = self.keyword_search(query, top_k=settings.top_k_retrieval)
        all_sources.extend(keyword_results)
        
        # 路径3：规则检索
        rule_results, matched_category = self.rule_based_search(query)
        all_sources.extend(rule_results)
        
        # 去重（基于内容）
        seen_contents = set()
        unique_sources = []
        for source in all_sources:
            content_hash = hash(source.content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_sources.append(source)
        
        print(f"📊 多路召回共返回 {len(unique_sources)} 条去重后的结果")
        
        # 重排
        if len(unique_sources) > top_k:
            final_sources = self.rerank(query, unique_sources, top_k)
        else:
            final_sources = unique_sources
        
        return final_sources
