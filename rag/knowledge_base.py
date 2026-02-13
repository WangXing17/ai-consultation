"""
知识库管理模块
负责：文档加载、清洗、切分、向量化、入库与增量更新
支持两种数据源：medical_knowledge.json（旧）、medical.txt（JSONL 病症库）
"""
import json
import uuid
import jieba
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from config import settings
from models import Document

# medical.txt 单条用于向量检索的文本最大长度（避免超长）
MEDICAL_CONTENT_MAX_LEN = 6000
# 单次插入 Milvus 的批次大小，避免 gRPC 消息超过 67MB 限制
MEDICAL_INSERT_BATCH_SIZE = 300

# 病症库各 VARCHAR 字段的 schema 最大长度，与加载时截断保持一致，避免插入报错
MEDICAL_FIELD_MAX_LEN = {
    "id": 100,
    "name": 256,
    "content": 65535,
    "category_primary": 256,
    "symptoms": 4096,
    "cure_department": 1024,
    "cure_way": 1024,
    "get_way": 1024,
    "cured_prob": 512,
}


class KnowledgeBase:
    """医疗知识库管理"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            model=settings.embedding_model
        )
        self.collection_name = settings.milvus_collection_name
        self.collection: Optional[Collection] = None
        self._connect_milvus()
        
    def _connect_milvus(self):
        """连接Milvus并创建collection"""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            
            # 检查collection是否存在
            if not utility.has_collection(self.collection_name):
                self._create_collection()
            
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"✅ 已连接到Milvus，collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Milvus连接失败: {e}")
            print("⚠️  请确保Milvus服务已启动")
    
    def _create_collection(self):
        """创建Milvus collection（旧版 JSON 结构：id, content, embedding, category, source）"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        ]
        
        schema = CollectionSchema(fields=fields, description="医疗知识库")
        collection = Collection(name=self.collection_name, schema=schema)
        
        # 创建向量索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ 已创建collection: {self.collection_name}")
    
    def _create_medical_collection(self):
        """
        创建病症库专用 Milvus collection（适配 medical.txt 结构）
        一病一条：id, name, content, embedding, category_primary, symptoms, cure_department, cure_way, get_way, cured_prob
        各 VARCHAR 长度与 MEDICAL_FIELD_MAX_LEN 一致
        """
        L = MEDICAL_FIELD_MAX_LEN
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=L["id"]),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=L["name"]),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=L["content"]),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            FieldSchema(name="category_primary", dtype=DataType.VARCHAR, max_length=L["category_primary"]),
            FieldSchema(name="symptoms", dtype=DataType.VARCHAR, max_length=L["symptoms"]),
            FieldSchema(name="cure_department", dtype=DataType.VARCHAR, max_length=L["cure_department"]),
            FieldSchema(name="cure_way", dtype=DataType.VARCHAR, max_length=L["cure_way"]),
            FieldSchema(name="get_way", dtype=DataType.VARCHAR, max_length=L["get_way"]),
            FieldSchema(name="cured_prob", dtype=DataType.VARCHAR, max_length=L["cured_prob"]),
        ]
        
        schema = CollectionSchema(fields=fields, description="病症库 medical.txt")
        collection = Collection(name=self.collection_name, schema=schema)
        
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 256}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ 已创建病症库 collection: {self.collection_name}")
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        加载文档
        支持JSON格式的医疗知识
        """
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                doc = Document(
                    id=item.get('id', str(uuid.uuid4())),
                    content=item.get('content', ''),
                    metadata={
                        'category': item.get('category', 'general'),
                        'source': item.get('source', 'unknown'),
                        'title': item.get('title', '')
                    }
                )
                documents.append(doc)
                
            print(f"✅ 加载了 {len(documents)} 条文档")
            return documents
        except FileNotFoundError:
            print(f"⚠️  文件不存在: {file_path}")
            return []
        except Exception as e:
            print(f"❌ 加载文档失败: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        # 去除多余空白
        text = ' '.join(text.split())
        # 可以添加更多清洗规则
        return text
    
    def split_documents(self, documents: List[Document], 
                       chunk_size: int = 500, 
                       chunk_overlap: int = 50) -> List[Document]:
        """
        切分文档
        使用RecursiveCharacterTextSplitter进行智能切分
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        split_docs = []
        for doc in documents:
            # 清洗文本
            cleaned_content = self.clean_text(doc.content)
            
            # 切分文本
            chunks = splitter.split_text(cleaned_content)
            
            for i, chunk in enumerate(chunks):
                split_doc = Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_id': i,
                        'parent_id': doc.id
                    }
                )
                split_docs.append(split_doc)
        
        print(f"✅ 切分为 {len(split_docs)} 个chunk")
        return split_docs
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """向量化文档"""
        texts = [doc.content for doc in documents]
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            print(f"✅ 向量化了 {len(documents)} 个文档")
            return documents
        except Exception as e:
            print(f"❌ 向量化失败: {e}")
            return documents
    
    def insert_documents(self, documents: List[Document]):
        """插入文档到Milvus"""
        if not self.collection:
            print("❌ Collection未初始化")
            return
        
        if not documents:
            print("⚠️  没有文档需要插入")
            return
        
        # 准备数据
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        categories = [doc.metadata.get('category', 'general') for doc in documents]
        sources = [doc.metadata.get('source', 'unknown') for doc in documents]
        
        # 插入数据
        try:
            entities = [ids, contents, embeddings, categories, sources]
            self.collection.insert(entities)
            self.collection.flush()
            print(f"✅ 成功插入 {len(documents)} 条文档到Milvus")
        except Exception as e:
            print(f"❌ 插入文档失败: {e}")
    
    def _build_medical_content(self, raw: Dict[str, Any]) -> str:
        """根据 medical.txt 单条 JSON 拼接用于向量检索的 content（名称+描述+症状+病因+预防+治疗等）"""
        parts = []
        name = (raw.get("name") or "").strip()
        desc = (raw.get("desc") or "").strip()
        symptom_list = raw.get("symptom") or []
        cause = (raw.get("cause") or "").strip()
        prevent = (raw.get("prevent") or "").strip()
        cure_way = raw.get("cure_way") or []
        check = raw.get("check") or []
        get_way = (raw.get("get_way") or "").strip()
        acompany = raw.get("acompany") or []
        
        if name:
            parts.append(f"疾病名称：{name}")
        if desc:
            parts.append(f"描述：{desc}")
        if symptom_list:
            parts.append("症状：" + "、".join(symptom_list))
        if cause:
            parts.append("病因：" + cause[:800])
        if prevent:
            parts.append("预防：" + prevent[:400])
        if cure_way:
            parts.append("治疗方式：" + "、".join(cure_way))
        if check:
            parts.append("检查：" + "、".join(check[:10]))
        if get_way:
            parts.append(f"传染/获得方式：{get_way}")
        if acompany:
            parts.append("并发症：" + "、".join(acompany))
        
        content = "\n".join(parts)
        if len(content) > MEDICAL_CONTENT_MAX_LEN:
            content = content[:MEDICAL_CONTENT_MAX_LEN] + "..."
        return self.clean_text(content)
    
    # raw 中 key 与 schema 字段名不一致时的映射（用于截断长度）
    _MEDICAL_RAW_KEY_TO_LEN = {"symptom": "symptoms"}
    
    def _medical_field_str(self, raw: Dict[str, Any], raw_key: str, default: str = "", *, list_join: str = "、") -> str:
        """从 raw 取出 raw_key 对应值，统一为字符串并截断到 schema 允许长度（列表用 list_join 连接）"""
        v = raw.get(raw_key)
        if v is None:
            s = default
        elif isinstance(v, list):
            s = list_join.join(str(x) for x in v)
        else:
            s = str(v)
        schema_key = self._MEDICAL_RAW_KEY_TO_LEN.get(raw_key, raw_key)
        max_len = MEDICAL_FIELD_MAX_LEN.get(schema_key, 512)
        return (s.strip() or default)[:max_len]
    
    def load_medical_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载 medical.txt（JSONL，每行一个病症 JSON）。
        所有 VARCHAR 字段按 MEDICAL_FIELD_MAX_LEN 截断，避免插入 Milvus 超长报错。
        返回 List[Dict]，每项包含 id, name, content, category_primary, symptoms, cure_department, cure_way, get_way, cured_prob
        """
        rows = []
        L = MEDICAL_FIELD_MAX_LEN
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    oid = raw.get("_id") or {}
                    if isinstance(oid, dict):
                        id_str = (oid.get("$oid") or str(uuid.uuid4()))[:L["id"]]
                    else:
                        id_str = str(oid)[:L["id"]]
                    
                    name = self._medical_field_str(raw, "name", "")[:L["name"]]
                    category_list = raw.get("category") or []
                    category_primary = (category_list[-1] if category_list else "其他")
                    category_primary = str(category_primary)[:L["category_primary"]]
                    symptoms = self._medical_field_str(raw, "symptom", "", list_join="、")[:L["symptoms"]]
                    cure_department = self._medical_field_str(raw, "cure_department", "", list_join="、")[:L["cure_department"]]
                    cure_way = self._medical_field_str(raw, "cure_way", "", list_join="、")[:L["cure_way"]]
                    get_way = self._medical_field_str(raw, "get_way", "无")[:L["get_way"]]
                    cured_prob = self._medical_field_str(raw, "cured_prob", "")[:L["cured_prob"]]
                    
                    content = self._build_medical_content(raw)
                    content = content[:L["content"]]
                    
                    rows.append({
                        "id": id_str,
                        "name": name,
                        "content": content,
                        "category_primary": category_primary,
                        "symptoms": symptoms,
                        "cure_department": cure_department,
                        "cure_way": cure_way,
                        "get_way": get_way,
                        "cured_prob": cured_prob,
                    })
            
            print(f"✅ 从 medical.txt 加载了 {len(rows)} 条病症")
            return rows
        except FileNotFoundError:
            print(f"⚠️  文件不存在: {file_path}")
            return []
        except Exception as e:
            print(f"❌ 加载 medical.txt 失败: {e}")
            return []
    
    def embed_medical_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对病症行的 content 做向量化，写入每行的 embedding 键"""
        texts = [r["content"] for r in rows]
        try:
            embeddings = self.embeddings.embed_documents(texts)
            for r, emb in zip(rows, embeddings):
                r["embedding"] = emb
            print(f"✅ 向量化了 {len(rows)} 条病症")
            return rows
        except Exception as e:
            print(f"❌ 向量化失败: {e}")
            return rows
    
    def insert_medical_rows(self, rows: List[Dict[str, Any]]):
        """将病症行分批插入当前 collection，避免 gRPC 单次消息超过 67MB 限制"""
        if not self.collection:
            print("❌ Collection 未初始化")
            return
        if not rows:
            print("⚠️  没有数据需要插入")
            return
        total = len(rows)
        batch_size = MEDICAL_INSERT_BATCH_SIZE
        inserted = 0
        try:
            for start in range(0, total, batch_size):
                batch = rows[start : start + batch_size]
                ids = [r["id"] for r in batch]
                names = [r["name"] for r in batch]
                contents = [r["content"] for r in batch]
                embeddings = [r["embedding"] for r in batch]
                category_primary = [r["category_primary"] for r in batch]
                symptoms = [r["symptoms"] for r in batch]
                cure_department = [r["cure_department"] for r in batch]
                cure_way = [r["cure_way"] for r in batch]
                get_way = [r["get_way"] for r in batch]
                cured_prob = [r["cured_prob"] for r in batch]
                entities = [ids, names, contents, embeddings, category_primary, symptoms, cure_department, cure_way, get_way, cured_prob]
                self.collection.insert(entities)
                inserted += len(batch)
                print(f"   已插入 {inserted}/{total} 条...")
            self.collection.flush()
            print(f"✅ 成功插入 {inserted} 条病症到 Milvus")
        except Exception as e:
            print(f"❌ 插入病症失败: {e}")
    
    def build_medical_knowledge_base(self, file_path: str):
        """
        使用 medical.txt 构建病症库：若已存在同名 collection 则先删除再创建新 schema，再加载、向量化、入库。
        """
        print("🚀 开始从 medical.txt 构建病症库...")
        try:
            connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)
        except Exception:
            pass
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"🗑️  已删除旧 collection: {self.collection_name}")
        self._create_medical_collection()
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
        rows = self.load_medical_txt(file_path)
        if not rows:
            return
        rows = self.embed_medical_rows(rows)
        self.insert_medical_rows(rows)
        self.collection.load()
        print("✅ 病症库构建完成！")
    
    def build_knowledge_base(self, file_path: str):
        """
        构建知识库完整流程（旧版 JSON 格式，如 medical_knowledge.json）
        1. 加载文档
        2. 清洗
        3. 切分
        4. 向量化
        5. 入库
        """
        print("🚀 开始构建知识库...")
        
        # 加载文档
        documents = self.load_documents(file_path)
        if not documents:
            return
        
        # 切分文档
        split_docs = self.split_documents(documents)
        
        # 向量化
        embedded_docs = self.embed_documents(split_docs)
        
        # 入库
        self.insert_documents(embedded_docs)
        
        print("✅ 知识库构建完成！")
    
    def incremental_update(self, documents: List[Document], update_type: str = "add"):
        """
        增量更新知识库
        支持：add/update/delete
        """
        print(f"🔄 执行增量更新，类型: {update_type}")
        
        if update_type == "delete":
            # 删除文档
            ids = [doc.id for doc in documents]
            expr = f"id in {ids}"
            self.collection.delete(expr)
            print(f"✅ 删除了 {len(ids)} 条文档")
        
        elif update_type in ["add", "update"]:
            if update_type == "update":
                # 先删除旧数据
                ids = [doc.id for doc in documents]
                expr = f"id in {ids}"
                self.collection.delete(expr)
            
            # 切分和向量化
            split_docs = self.split_documents(documents)
            embedded_docs = self.embed_documents(split_docs)
            
            # 插入新数据
            self.insert_documents(embedded_docs)
        
        print("✅ 增量更新完成")


if __name__ == "__main__":
    # 测试知识库构建
    kb = KnowledgeBase()
    # kb.build_knowledge_base("data/medical_knowledge.json")
