#!/bin/bash

echo "🔍 检查并安装缺失的依赖包..."
echo ""

# 你已经安装的包：
# - langchain (1.2.6)
# - langchain-openai (1.1.7)
# - langchain-community (0.4.1)
# - openai (2.15.0)
# - redis (7.1.0)
# - pydantic (2.12.5)
# - python-dotenv (1.2.1)
# - httpx (0.28.1)

echo "✅ 已有的包："
echo "  - langchain (1.2.6)"
echo "  - langchain-openai (1.1.7)"
echo "  - openai (2.15.0)"
echo "  - redis (7.1.0)"
echo "  - pydantic (2.12.5)"
echo "  - python-dotenv (1.2.1)"
echo "  - httpx (0.28.1)"
echo ""

echo "📦 需要安装的包："
echo "  - fastapi"
echo "  - uvicorn"
echo "  - python-multipart"
echo "  - pymilvus"
echo "  - jieba"
echo "  - rank-bm25"
echo ""

read -p "是否继续安装？(y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "开始安装..."
    pip install fastapi uvicorn[standard] python-multipart pymilvus jieba rank-bm25
    echo ""
    echo "✅ 安装完成！"
fi
