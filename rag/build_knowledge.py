#!/usr/bin/env python
"""
知识库构建脚本
使用方法：python build_knowledge.py
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base import KnowledgeBase


def main():
    print("=" * 60)
    print("  RAG智能问诊助手 - 知识库构建工具")
    print("=" * 60)
    print()
    
    # 知识文件路径
    knowledge_file = "data/medical_knowledge.json"
    
    print(f"📚 准备构建知识库...")
    print(f"📄 知识文件: {knowledge_file}")
    print()
    
    try:
        # 创建知识库实例
        kb = KnowledgeBase()
        
        # 构建知识库
        kb.build_knowledge_base(knowledge_file)
        
        print()
        print("=" * 60)
        print("🎉 知识库构建成功！")
        print("=" * 60)
        print()
        print("下一步：启动服务")
        print("  python main.py")
        print()
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到知识文件 {knowledge_file}")
        print(f"   请确保文件存在")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ 错误：{e}")
        print()
        print("常见问题排查：")
        print("1. Milvus服务是否启动？")
        print("   检查：docker ps | grep milvus")
        print()
        print("2. OpenAI API Key是否配置？")
        print("   检查 .env 文件中的 OPENAI_API_KEY")
        print()
        print("3. 网络连接是否正常？")
        sys.exit(1)


if __name__ == "__main__":
    main()
