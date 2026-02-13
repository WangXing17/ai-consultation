# medical.txt 数据结构说明

## 文件概况

- **格式**：JSONL（每行一个 JSON 对象）
- **总条数**：约 8808 条
- **来源**：类 MongoDB 导出（含 `_id.$oid`）

---

## 前 10 条病症摘要

| # | 疾病名称 | 科室 | 症状数 | 传染方式 | 治愈概率 | 饮食/药物 |
|---|----------|------|--------|----------|----------|------------|
| 1 | 肺泡蛋白质沉积症 | 内科/呼吸内科 | 5 | 无传染性 | 约40% | 无 |
| 2 | 百日咳 | 儿科/小儿内科 | 8 | 呼吸道传播 | 98% | 有 |
| 3 | 苯中毒 | 急诊科 | 3 | 无传染性 | 75% | 有 |
| 4 | 喘息样支气管炎 | 内科/呼吸内科 | 8 | 无传染性 | 95% | 有 |
| 5 | 成人呼吸窘迫综合征 | 内科/呼吸内科 | 3 | 无传染性 | 85% | 有 |
| 6 | 大量羊水吸入 | 儿科/小儿内科 | 7 | 无传染性 | 85% | 无 |
| 7 | 单纯性肺嗜酸粒细胞浸润症 | 内科/呼吸内科 | 6 | 无传染性 | 100% | 有 |
| 8 | 大叶性肺炎 | 内科/呼吸内科 | 6 | 无传染性 | 90%以上 | 有 |
| 9 | 大楼病综合征 | 其他科室/其他综合 | 9 | 无传染性 | 90% | 有 |
| 10 | 二硫化碳中毒 | 急诊科 | 8 | 无传染性 | 80-85% | 有 |

---

## 字段说明（按是否必选）

### 必有字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` | `{"$oid": "..."}` | MongoDB 主键，用作 Milvus 主键 |
| `name` | string | 疾病名称 |
| `desc` | string | 疾病描述（长度不一，部分很长） |
| `category` | array[string] | 分类，如 `["疾病百科","内科","呼吸内科"]` |
| `symptom` | array[string] | 症状列表 |
| `cure_department` | array[string] | 就诊科室 |
| `cure_way` | array[string] | 治疗方式 |
| `get_way` | string | 传染/获得方式 |
| `cured_prob` | string | 治愈概率 |
| `cost_money` | string | 费用说明 |
| `check` | array[string] | 检查项目 |
| `prevent` | string | 预防 |
| `cause` | string | 病因（可能很长） |
| `yibao_status` | string | 是否医保 |
| `get_prob` | string | 患病概率 |
| `acompany` | array[string] | 并发症 |
| `cure_lasttime` | string | 治疗周期 |
| `recommand_drug` | array | 推荐药物 |
| `drug_detail` | array | 药物详情 |

### 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `easy_get` | string | 易感人群 |
| `common_drug` | array[string] | 常用药 |
| `do_eat` | array[string] | 宜吃 |
| `not_eat` | array[string] | 忌吃 |
| `recommand_eat` | array[string] | 推荐食谱 |

---

## Milvus 设计结论（已实现）

1. **一病一条向量**：每条病症一条记录，`content` 为拼接后的检索文本（名称+描述+症状+病因+预防等），用于做向量检索。
2. **主键**：使用 `_id.$oid` 作为 Milvus 主键，保证唯一且与源数据一致。
3. **标量字段**：保留 `name`、`category_primary`、`symptoms`、`cure_department`、`cure_way`、`get_way`、`cured_prob`，便于过滤与展示。
4. **长文本**：`desc`/`cause` 等拼接进 `content` 时截断为 ≤ 6000 字（见 `knowledge_base.MEDICAL_CONTENT_MAX_LEN`）。

### 实际 Milvus Schema（病症库）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(100) | 主键，来自 `_id.$oid` |
| name | VARCHAR(256) | 疾病名称 |
| content | VARCHAR(65535) | 拼接后的检索文本 |
| embedding | FLOAT_VECTOR(dim) | 向量 |
| category_primary | VARCHAR(128) | 主科室，如「呼吸内科」 |
| symptoms | VARCHAR(2048) | 症状，顿号分隔 |
| cure_department | VARCHAR(512) | 就诊科室 |
| cure_way | VARCHAR(512) | 治疗方式 |
| get_way | VARCHAR(128) | 传染/获得方式 |
| cured_prob | VARCHAR(64) | 治愈概率 |

### 构建方式

- **命令行**：`python build_medical.py` 或 `python build_medical.py data/medical.txt`
- **API**：`POST /api/knowledge/build_medical?file_path=data/medical.txt`
