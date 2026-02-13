"""
RAG 评估脚本：使用 RAGAS 对医疗问答 RAG 系统进行自动化评估。
运行前请确保：1）知识库已构建 2）已配置 OPENAI_API_KEY 等环境变量。
用法：
  cd rag && python evaluation.py
  python evaluation.py --data data/eval_questions.json --output results/eval_result.json
"""
import json
import argparse
import sys
from pathlib import Path

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 必须在导入任何 langchain/ragas 之前执行：新版 langchain-core 不再在 output_parsers 根下导出 PydanticOutputParser，导致 ragas 报错
def _patch_pydantic_output_parser():
    import langchain_core.output_parsers as _lc_parsers
    if hasattr(_lc_parsers, "PydanticOutputParser"):
        return
    for _import_fn in (
        lambda: __import__("langchain_core.output_parsers.pydantic", fromlist=["PydanticOutputParser"]).PydanticOutputParser,
        lambda: __import__("langchain.output_parsers", fromlist=["PydanticOutputParser"]).PydanticOutputParser,
    ):
        try:
            _lc_parsers.PydanticOutputParser = _import_fn()
            return
        except (ImportError, AttributeError):
            continue
    import warnings
    warnings.warn(
        "未能注入 PydanticOutputParser，若后续 ragas 报错，请执行: pip install 'langchain-core>=0.2.0,<0.3.0'",
        UserWarning,
    )

_patch_pydantic_output_parser()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Embeddings 供 RAGAS 指标使用

from config import settings
from retriever import MultiPathRetriever
from query_optimizer import optimize as optimize_query
from models import KnowledgeSource


def build_prompt_for_eval(question: str, knowledge_sources: list) -> tuple:
    """与 main.build_prompt 一致的构建逻辑（无历史），供评估脚本使用"""
    knowledge_text = ""
    for i, source in enumerate(knowledge_sources, 1):
        source_type = "【知识库】" if getattr(source, "source", "") == "knowledge_base" else "【联网搜索】"
        content = getattr(source, "content", "") or ""
        knowledge_text += f"\n{source_type} 来源{i}：\n{content}\n"
    system_prompt = """你是一个专业的医疗问诊助手，具备丰富的医学知识。你的任务是：
1. **理解病情**：仔细分析用户的症状描述
2. **信息补全**：如果信息不完整，主动询问关键信息
3. **知识检索**：基于提供的医疗知识，给出专业建议
4. **结构化建议**：提供清晰的分步建议
**回答要求**：专业、准确、易懂；引用知识来源时标注【知识库】或【联网搜索】；给出3-5条结构化建议；必要时提醒用户就医。
**重要提示**：你不能替代专业医生诊断；紧急情况请立即就医；建议仅供参考。"""
    user_prompt = f"""用户问题：{question}\n\n参考知识：\n{knowledge_text}\n请基于以上知识给出专业的问诊建议。"""
    return system_prompt, user_prompt


def run_rag_pipeline(question: str, retriever: MultiPathRetriever, llm: ChatOpenAI) -> tuple:
    """对单条问题跑一遍 RAG：检索 + 生成，返回 (answer, contexts)。评估时不走 MCP 兜底以保持可复现。"""
    history = []
    retrieval_query = optimize_query(
        question,
        history=history,
        enable_rewrite=settings.enable_query_rewrite,
        enable_normalize=settings.enable_query_normalize,
    )
    knowledge_sources = retriever.retrieve(retrieval_query)
    contexts = [getattr(s, "content", "") or "" for s in knowledge_sources]
    if not contexts:
        contexts = [""]
    system_prompt, user_prompt = build_prompt_for_eval(question, knowledge_sources)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    answer = (response.content or "").strip()
    return answer, contexts


def load_eval_data(path: str) -> list:
    """加载评估数据：每项需包含 question，可选 ground_truth。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评估数据文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def collect_rag_samples(eval_data: list, retriever: MultiPathRetriever, llm: ChatOpenAI) -> list:
    """对每条评估问题跑 RAG，收集 RAGAS 所需的样本（user_input, retrieved_contexts, response, reference）。"""
    samples = []
    for i, item in enumerate(eval_data):
        question = (item.get("question") or "").strip()
        if not question:
            continue
        ground_truth = (item.get("ground_truth") or "").strip() or None
        print(f"  [{i+1}/{len(eval_data)}] 运行 RAG: {question[:50]}...")
        try:
            answer, contexts = run_rag_pipeline(question, retriever, llm)
        except Exception as e:
            print(f"    ⚠️ 失败: {e}")
            answer, contexts = "", [""]
        sample = {
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth:
            sample["reference"] = ground_truth
        samples.append(sample)
    return samples


def run_ragas_evaluation(samples: list, output_path: str = None) -> dict:
    """使用 RAGAS 对样本打分。"""
    try:
        from ragas import evaluate, EvaluationDataset
    except ImportError as e:
        print("请先安装 ragas 与依赖: pip install ragas pandas datasets anthropic")
        raise e

    # evaluate() 只接受「已初始化的 metric 实例」且与 LangchainLLMWrapper/LangchainEmbeddingsWrapper 兼容；弃用警告已忽略
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from ragas.metrics import Faithfulness, ContextPrecision, AnswerRelevancy
        except ImportError:
            from ragas.metrics.collections import Faithfulness, ContextPrecision, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        model=settings.openai_model,
        temperature=0.1,
    ))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        model=settings.embedding_model,
    ))

    # 已初始化的 metric 实例列表（evaluate 要求 list of metric instances）
    metric_instances = [
        Faithfulness(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    if not samples:
        print("没有有效样本，跳过 RAGAS 评估")
        return {}

    # 构建 RAGAS 数据集（使用 SingleTurnSample 字段名）
    eval_dataset = EvaluationDataset.from_list(samples)

    print("运行 RAGAS 评估...")
    result = evaluate(dataset=eval_dataset, metrics=metric_instances)

    # 汇总分数（兼容不同 ragas 版本的 result 结构）
    summary = {}
    scores_list = []
    if hasattr(result, "scores") and result.scores:
        scores_list = result.scores
    if hasattr(result, "to_pandas"):
        try:
            scores_df = result.to_pandas()
            score_cols = [c for c in scores_df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]
            for col in score_cols:
                try:
                    if scores_df[col].dtype in ("float64", "float32"):
                        summary[col] = float(scores_df[col].mean())
                except Exception:
                    pass
            scores_list = scores_df.to_dict(orient="records")
        except Exception as e:
            print(f"to_pandas 失败: {e}")
    if not summary and scores_list:
        # result.scores 为 list of dict
        for d in scores_list:
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    summary[k] = summary.get(k, []) + [v]
        summary = {k: sum(v) / len(v) for k, v in summary.items()}

    if summary:
        print("\n========== RAGAS 评估结果 ==========")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "scores_per_sample": scores_list}, f, ensure_ascii=False, indent=2)
            print(f"\n结果已写入: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="RAG 医疗问答系统 RAGAS 评估")
    parser.add_argument("--data", default="data/eval_questions.json", help="评估问题 JSON 路径")
    parser.add_argument("--output", default="", help="评估结果输出 JSON 路径（不填则只打印）")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path
    output_path = args.output
    if output_path and not Path(output_path).is_absolute():
        output_path = str(Path(__file__).resolve().parent / output_path)

    print("加载评估数据...")
    eval_data = load_eval_data(str(data_path))
    print(f"共 {len(eval_data)} 条问题")

    print("初始化检索器与 LLM...")
    retriever = MultiPathRetriever()
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        model=settings.openai_model,
        temperature=0.7,
    )

    print("运行 RAG 并收集样本...")
    samples = collect_rag_samples(eval_data, retriever, llm)
    print(f"有效样本数: {len(samples)}")

    if not samples:
        print("无有效样本，退出")
        return

    summary = run_ragas_evaluation(samples, output_path=output_path or None)
    if summary:
        print("\n评估完成。")


if __name__ == "__main__":
    main()
