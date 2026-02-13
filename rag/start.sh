#!/bin/bash

# RAG智能问诊助手 - 启动脚本

echo "=========================================="
echo "  RAG智能问诊助手 - 启动检查"
echo "=========================================="
echo ""

# 检查Python版本
echo "🐍 检查Python版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python版本: $python_version"
echo ""

# 检查.env文件
echo "📝 检查配置文件..."
if [ ! -f .env ]; then
    echo "   ⚠️  .env文件不存在"
    echo "   正在复制.env.example..."
    cp .env.example .env
    echo "   ✅ 已创建.env文件，请编辑配置后重新运行"
    echo ""
    echo "   至少需要配置："
    echo "   - OPENAI_API_KEY"
    echo "   - MILVUS_HOST 和 MILVUS_PORT"
    echo "   - REDIS_HOST 和 REDIS_PORT"
    exit 1
else
    echo "   ✅ .env文件存在"
fi
echo ""

# 检查依赖包
echo "📦 检查依赖包..."
missing_packages=()

# 检查必需的包
for package in fastapi uvicorn pymilvus jieba rank-bm25; do
    if ! python -c "import $package" 2>/dev/null; then
        missing_packages+=($package)
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "   ⚠️  缺少以下依赖包: ${missing_packages[*]}"
    echo ""
    read -p "   是否现在安装？(y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   正在安装..."
        pip install fastapi uvicorn[standard] python-multipart pymilvus jieba rank-bm25
        echo "   ✅ 安装完成"
    else
        echo "   请先安装依赖包："
        echo "   pip install fastapi uvicorn[standard] python-multipart pymilvus jieba rank-bm25"
        exit 1
    fi
else
    echo "   ✅ 所有依赖包已安装"
fi
echo ""

# 检查Milvus连接
echo "🗄️  检查Milvus连接..."
if nc -z localhost 19530 2>/dev/null; then
    echo "   ✅ Milvus服务正在运行 (localhost:19530)"
else
    echo "   ⚠️  无法连接到Milvus (localhost:19530)"
    echo "   请确保Milvus服务已启动："
    echo "   docker-compose up -d"
    echo ""
    read -p "   是否继续启动？(服务可能无法正常工作) (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# 检查Redis连接
echo "💾 检查Redis连接..."
if nc -z localhost 6379 2>/dev/null; then
    echo "   ✅ Redis服务正在运行 (localhost:6379)"
else
    echo "   ⚠️  无法连接到Redis (localhost:6379)"
    echo "   系统将继续运行，但缓存功能将不可用"
    echo "   建议启动Redis："
    echo "   docker run -d --name redis-rag -p 6379:6379 redis:latest"
fi
echo ""

# 检查知识库
echo "📚 检查知识库..."
if python -c "from pymilvus import connections, utility; connections.connect(host='localhost', port='19530'); has = utility.has_collection('medical_knowledge'); print('exists' if has else 'not_exists')" 2>/dev/null | grep -q "exists"; then
    echo "   ✅ 知识库已构建"
else
    echo "   ⚠️  知识库未构建"
    echo ""
    read -p "   是否现在构建知识库？(y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   正在构建知识库..."
        python -c "from knowledge_base import KnowledgeBase; kb = KnowledgeBase(); kb.build_knowledge_base('data/medical_knowledge.json')"
        echo "   ✅ 知识库构建完成"
    else
        echo "   警告：没有知识库，系统可能无法正常工作"
        echo "   稍后可以运行："
        echo "   python -c \"from knowledge_base import KnowledgeBase; kb = KnowledgeBase(); kb.build_knowledge_base('data/medical_knowledge.json')\""
    fi
fi
echo ""

# 启动服务
echo "=========================================="
echo "🚀 启动RAG智能问诊助手..."
echo "=========================================="
echo ""
echo "访问地址："
echo "  - API文档: http://localhost:8000/docs"
echo "  - 健康检查: http://localhost:8000/"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python main.py
