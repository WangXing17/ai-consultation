"""
测试客户端
演示如何调用问诊API
"""
import sys
import os
import asyncio
import httpx
import json

# 添加当前目录到Python路径（如果需要导入本地模块）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# SSE流式请求示例
async def test_stream():
    """测试SSE流式接口"""
    url = "http://localhost:8000/api/consult/stream"
    
    request_data = {
        "question": "我最近总是头痛，特别是太阳穴位置，该怎么办？",
        "user_id": "test_user_001",
        "history": []
    }
    
    print("🔍 发送问诊请求（流式）...")
    print(f"问题: {request_data['question']}\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=request_data) as response:
            print("📡 接收流式响应:\n")
            print("-" * 80)
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    
                    try:
                        data = json.loads(data_str)
                        msg_type = data.get("type")
                        
                        if msg_type == "status":
                            print(f"[状态] {data['message']}")
                        
                        elif msg_type == "sources":
                            print("\n[知识来源]")
                            for i, source in enumerate(data['sources'], 1):
                                source_type = "知识库" if source['source'] == "knowledge_base" else "联网搜索"
                                print(f"{i}. [{source_type}] {source['content'][:100]}...")
                            print()
                        
                        elif msg_type == "content":
                            print(data['content'], end='', flush=True)
                        
                        elif msg_type == "suggestions":
                            print("\n\n[结构化建议]")
                            for i, suggestion in enumerate(data['suggestions'], 1):
                                print(f"{i}. {suggestion}")
                        
                        elif msg_type == "done":
                            print("\n\n✅ 问诊完成")
                        
                        elif msg_type == "error":
                            print(f"\n❌ 错误: {data['message']}")
                        
                        elif msg_type == "cached":
                            print(f"💾 {data['message']}")
                    
                    except json.JSONDecodeError:
                        pass
            
            print("-" * 80)


# 非流式请求示例
async def test_normal():
    """测试普通接口"""
    url = "http://localhost:8000/api/consult"
    
    request_data = {
        "question": "发烧到39度了怎么办？",
        "user_id": "test_user_002",
        "history": []
    }
    
    print("🔍 发送问诊请求（非流式）...")
    print(f"问题: {request_data['question']}\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=request_data)
        result = response.json()
        
        print("📡 问诊结果:\n")
        print("-" * 80)
        print(f"\n回答:\n{result['answer']}\n")
        
        if result['sources']:
            print("\n知识来源:")
            for i, source in enumerate(result['sources'], 1):
                source_type = "知识库" if source['source'] == "knowledge_base" else "联网搜索"
                print(f"{i}. [{source_type}] {source['content'][:100]}...")
        
        if result['suggestions']:
            print("\n结构化建议:")
            for i, suggestion in enumerate(result['suggestions'], 1):
                print(f"{i}. {suggestion}")
        
        print("-" * 80)


async def main():
    """主函数"""
    print("=" * 80)
    print("RAG智能问诊助手 - 测试客户端")
    print("=" * 80)
    print()
    
    # 测试SSE流式接口
    await test_stream()
    
    print("\n\n")
    
    # 测试普通接口
    await test_normal()


if __name__ == "__main__":
    asyncio.run(main())
