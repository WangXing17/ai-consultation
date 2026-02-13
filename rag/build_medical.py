#!/usr/bin/env python
"""
从 data/medical.txt（JSONL 病症数据）构建病症库。
会删除已有同名 collection，按新 schema 创建并写入。
使用方法：python build_medical.py [文件路径]
默认文件路径：data/medical.txt
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base import KnowledgeBase
from config import settings


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/medical.txt"
    
    print("=" * 60)
    print("  RAG智能问诊助手 - 病症库构建（medical.txt）")
    print("=" * 60)
    print()
    print(f"📄 数据文件: {file_path}")
    print()
    
    if not os.path.isfile(file_path):
        print(f"❌ 错误：文件不存在 {file_path}")
        sys.exit(1)
    
    try:
        kb = KnowledgeBase()
        kb.build_medical_knowledge_base(file_path)
        print()
        print("🎉 病症库构建完成！可启动服务：python main.py")
    except Exception as e:
        print(f"❌ 错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
