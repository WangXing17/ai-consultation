# RAGAS 评估接入说明

使用 [RAGAS](https://docs.ragas.io/) 对 RAG 医疗问答系统进行自动化评估，从**检索质量**和**生成质量**两方面打分。

## 指标含义

| 指标 | 含义 |
|------|------|
| **Faithfulness（忠实度）** | 回答中的陈述有多少能被检索到的上下文支持，越高越少“幻觉” |
| **Context Precision（上下文精确度）** | 检索到的内容中，相关片段是否排在前面，衡量检索排序质量 |
| **Answer Relevancy（答案相关性）** | 回答与问题的相关程度，是否切题、是否冗余或缺失 |

可选（需提供标准答案 `ground_truth`）：

- **Context Recall**：检索到的上下文覆盖标准答案的程度

## 环境准备

1. 安装依赖（已写入 `requirements.txt`）：
   ```bash
   pip install ragas pandas datasets
   ```

2. 确保已配置 `OPENAI_API_KEY`（及 `OPENAI_API_BASE` 若使用代理），RAGAS 的指标会调用 LLM/Embedding。

3. 确保知识库已构建、Milvus 可用，评估时会真实跑检索与生成。

## 测试集格式

默认使用 `data/eval_questions.json`，每项包含：

- **question**（必填）：待评估的问题
- **ground_truth**（可选）：标准答案，用于 Context Recall 等需要参考的指标

示例：

```json
[
  {
    "question": "我发烧39度了怎么办？",
    "ground_truth": "发热39℃属于高热，建议物理降温并服用退热药..."
  }
]
```

可自行增删、修改问题，或新建 JSON 文件。

## 运行评估

在项目 `rag` 目录下执行：

```bash
cd rag

# 使用默认测试集，仅打印结果
python evaluation.py

# 指定测试集与结果输出路径
python evaluation.py --data data/eval_questions.json --output results/eval_result.json
```

脚本会：

1. 加载测试问题
2. 对每条问题执行一次完整 RAG 流程（提问优化 → 检索 → 生成），**不调用 MCP 联网**，保证可复现
3. 用 RAGAS 计算各指标
4. 在终端打印各指标均值，若指定 `--output` 则写入 JSON（含逐条分数）

## 结果示例

终端输出示例：

```
========== RAGAS 评估结果 ==========
  faithfulness: 0.8523
  context_precision: 0.7801
  answer_relevancy: 0.8912

结果已写入: results/eval_result.json
```

输出 JSON 结构：

- **summary**：各指标平均值
- **scores_per_sample**：每条问题的各指标分数，便于分析 Bad Case

## 与 CI / 迭代优化结合

- 在 CI 中定期跑 `evaluation.py`，对 `summary` 做阈值或趋势判断（如 faithfulness 低于 0.7 则失败）。
- 调整检索（top_k、重排）、提示词或 Query 优化后，重新跑评估对比 `summary` 与 `scores_per_sample`，量化改进效果。

## 常见问题

- **ImportError: cannot import Faithfulness / ContextPrecision**  
  不同 RAGAS 版本路径可能不同，脚本已做 `ragas.metrics` 与 `ragas.metrics.collections` 的兼容；若仍报错，可执行 `pip install -U ragas` 后重试。

- **评估很慢**  
  每条样本会执行 1 次检索 + 1 次生成 + 多次 RAGAS 内部 LLM 调用，样本多时耗时会较长，可先用少量样本（如 5 条）验证流程。

- **希望评估时也走 MCP 兜底**  
  当前为保持可复现性，评估脚本未调用 `mcp_manager.enhance_retrieval`。若需包含联网结果，可在 `evaluation.py` 的 `run_rag_pipeline` 中自行接入 MCP 逻辑（需异步或同步封装一致）。
