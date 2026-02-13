# 快速启动指南

## 📦 第一步：安装缺失的依赖

你已经安装了大部分依赖，只需要安装以下几个包：

```bash
pip install fastapi uvicorn[standard] python-multipart pymilvus jieba rank-bm25
```

或者使用最小化依赖文件：

```bash
pip install -r requirements_minimal.txt
```

## ⚙️ 第二步：配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，至少配置以下内容：

```env
# OpenAI配置（必需）
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002

# Milvus配置（必需）
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis配置（必需）
REDIS_HOST=localhost
REDIS_PORT=6379

# Bing搜索配置（可选，用于MCP兜底）
BING_SEARCH_API_KEY=your_bing_key_here
```

## 🐳 第三步：启动依赖服务

### 方式1：使用Docker（推荐）

**启动Milvus：**
```bash
# 创建一个临时目录用于docker-compose
mkdir -p ~/milvus_data
cd ~/milvus_data

# 下载docker-compose配置
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动Milvus
docker-compose up -d

# 查看状态
docker-compose ps
```

**启动Redis：**
```bash
docker run -d --name redis-rag -p 6379:6379 redis:latest

# 验证Redis
docker ps | grep redis
```

### 方式2：使用本地服务

如果你已经有本地的Milvus和Redis服务，确保它们正在运行即可。

**验证Milvus：**
```bash
# 检查端口是否开放
nc -zv localhost 19530
```

**验证Redis：**
```bash
# 检查端口是否开放
nc -zv localhost 6379

# 或使用redis-cli
redis-cli ping
# 应该返回 PONG
```

## 📚 第四步：构建知识库

返回项目目录并构建知识库：

```bash
cd /Users/mengzhifang/Mypro/ai/ai-chat/rag

python -c "from knowledge_base import KnowledgeBase; kb = KnowledgeBase(); kb.build_knowledge_base('data/medical_knowledge.json')"
```

输出示例：
```
✅ 已连接到Milvus，collection: medical_knowledge
🚀 开始构建知识库...
✅ 加载了 20 条文档
✅ 切分为 45 个chunk
✅ 向量化了 45 个文档
✅ 成功插入 45 条文档到Milvus
✅ 知识库构建完成！
```

## 🚀 第五步：启动服务

```bash
python main.py
```

看到以下输出表示启动成功：
```
🚀 启动RAG智能问诊助手...
✅ Redis连接成功
✅ BM25索引构建完成，包含 45 个文档
✅ 系统启动完成
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 🧪 第六步：测试接口

### 方式1：使用测试客户端（推荐）

打开新终端，运行：

```bash
cd /Users/mengzhifang/Mypro/ai/ai-chat/rag
python test_client.py
```

### 方式2：使用curl

**测试流式接口：**
```bash
curl -N -X POST http://localhost:8000/api/consult/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "我发烧39度了怎么办？",
    "user_id": "test_001"
  }'
```

**测试普通接口：**
```bash
curl -X POST http://localhost:8000/api/consult \
  -H "Content-Type: application/json" \
  -d '{
    "question": "头痛应该怎么处理？",
    "user_id": "test_002"
  }' | jq
```

### 方式3：浏览器访问API文档

打开浏览器访问：`http://localhost:8000/docs`

可以看到交互式API文档，直接在浏览器中测试。

## 📊 测试问题示例

1. **常规症状**：
   - "我发烧39度了怎么办？"
   - "最近总是头痛，特别是太阳穴位置"
   - "咳嗽一直不好，有痰"

2. **疾病管理**：
   - "高血压患者日常应该注意什么？"
   - "糖尿病如何控制血糖？"

3. **紧急情况**：
   - "胸口剧烈疼痛，该怎么办？"
   - "中暑了怎么急救？"

4. **用药咨询**：
   - "发烧吃什么退烧药？"
   - "抗生素应该怎么用？"

## 🔧 常见问题

### 1. Milvus连接失败

**错误**：`❌ Milvus连接失败`

**解决**：
- 检查Milvus是否启动：`docker ps | grep milvus`
- 检查端口：`lsof -i :19530`
- 查看日志：`docker logs milvus-standalone`

### 2. Redis连接失败

**错误**：`⚠️  Redis连接失败`

**解决**：
- 检查Redis是否启动：`docker ps | grep redis`
- 测试连接：`redis-cli ping`
- 系统会继续运行，但不会有缓存功能

### 3. OpenAI API错误

**错误**：`openai.AuthenticationError`

**解决**：
- 检查 `.env` 中的 `OPENAI_API_KEY` 是否正确
- 检查 `OPENAI_API_BASE` 是否正确（如果使用国内代理）

### 4. 知识库构建失败

**错误**：向量化或插入失败

**解决**：
- 确保Milvus正常运行
- 检查OpenAI API配额
- 可以分批次处理大量文档

## 📝 下一步

1. **添加更多医疗知识**：
   - 编辑 `data/medical_knowledge.json`
   - 使用增量更新API添加新知识

2. **自定义配置**：
   - 调整 `.env` 中的检索参数
   - 修改 `TOP_K_RETRIEVAL` 和 `TOP_K_RERANK`

3. **集成到你的应用**：
   - 使用 `/api/consult/stream` 进行流式问诊
   - 使用 `/api/consult` 进行普通问诊

4. **监控和优化**：
   - 观察检索命中率
   - 调整相似度阈值
   - 优化prompt模板

## 🎉 完成！

现在你的RAG智能问诊助手已经可以使用了！

有任何问题请查看 `README.md` 或提出Issue。
