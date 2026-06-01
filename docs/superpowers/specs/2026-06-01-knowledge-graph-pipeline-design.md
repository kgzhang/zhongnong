# 猪抗生素替代物知识图谱 — 文献数据抽取管线设计

> 日期: 2026-06-01 | 状态: 待审阅

## 1. 项目目标

从 PubMed/PMC 文献全文中，按照 BACKGROUND.md 规定的五大模块抽取规则，深度结构化提取猪抗生素替代物的知识图谱数据。图谱遵循 SCHEMA.tsv 定义的 13 类实体与 6 类关系，替代物实体对齐 ALTERNATIVE.tsv 分类体系。

## 2. 核心约束

| 维度 | 决策 |
|------|------|
| 抽取模式 | LLM 初抽 + 人工复核 |
| 文献量 | 1000-5000+ 篇 |
| 输入格式 | PMC/NLM XML（已下载完备） |
| LLM | 商业 API（GPT-4 / Claude API） |
| 技术栈 | Python（已有 `.venv`） |
| 交付物 | 结构化 TSV 表格（人工审查） + Neo4j Cypher 脚本（最终载体） |

## 3. 整体架构 — 四阶段批次流水线

```
阶段0 (程序化)        阶段1 (程序化)      阶段2 (LLM json_object)    阶段3 (LLM json_object)    阶段4 (程序化)

PubMed API            XML Parser         模块C: 替代物实体           Pass1: 全量结果抽取           Schema校验
   │                     │                   │                         │                          │
   ▼                     ▼                   ▼                         ▼                          ▼
检索+排重    ──→     章节结构化    ──→    模块A: 试验设计     ──→   Pass2: 对齐+关系    ──→     业务规则校验
   │                     │                   │                         │                          │
   ▼                     ▼                   ▼                         ▼                          ▼
总池清单             structured_         模块B: 指标体系             Result实体                   人工闸门
                    sections/               │                       关系链                         │
                                            ▼                                                    ▼
                                       entities/                                          TSV + Neo4j
                                       (13类实体)                                          双轨导出
```

**LLM 调用仅发生在阶段 2 和阶段 3**（图谱提取核心环节）。阶段 0、1、4 均为确定性程序逻辑。

## 4. 阶段 0：文献库扩充与排重（程序化）

### 输入
- `ALTERNATIVE.tsv` 中全部物质名称
- 已有 PDF/XML 库存的 DOI 清单

### 流程
1. 以每个物质名 + 猪限定词（pig/swine/piglet）构造 PubMed E-utilities 检索式
2. 无年份限制回溯检索，获取 PMID/DOI/title/journal/year/abstract/PMCID
3. 以 DOI 为唯一键，与新检索结果排重
4. 标记无 XML 全文的文献（仅有摘要）
5. 输出 `literature_pool.tsv`（DOI + PMCID + XML 本地路径映射 + 检索来源词）

### 输出
- `literature_pool.tsv`：全量待抽取文献总池
- `missing_fulltext.tsv`：仅有摘要、缺少 XML 全文的文献清单

## 5. 阶段 1：XML 结构化解析（程序化）

### 输入
全量 PMC/NLM XML 文件

### 流程
1. **元数据提取**：解析 `<article-meta>` → DOI, PMID, title, journal, year, abstract
2. **结论句提取**：定位 `<abstract>` 最后 1-3 句，匹配结论提示词（In conclusion / These results suggest / Overall / Therefore / In summary / Collectively），填入 `Literature.abstract_conclusion`
3. **章节切分**：解析 `<body>` → `<sec>` 章节树，映射到 M&M / Results / Discussion
4. **表格提取**：解析 `<table-wrap>` → 结构化表格信息
5. 输出逐文献 JSON：`structured_sections/{DOI}.json`

### 输出
`structured_sections/{DOI}.json`，结构：
```json
{
  "doi": "10.1016/xxx",
  "pmid": "36789012",
  "sections": {
    "abstract": { "start_offset": 0, "end_offset": 1250, "conclusion_sentence": "..." },
    "materials_and_methods": { "start_offset": 1251, "end_offset": 8420, "subsections": [...] },
    "results": { "start_offset": 8421, "end_offset": 15400, "subsections": [...] },
    "discussion": { "start_offset": 15401, "end_offset": 22000 }
  },
  "tables": [{ "table_id": "Table 1", "title": "...", "content": "..." }],
  "confidence": 0.95, "warnings": []
}
```

### 质量闸门
- 章节标题模糊匹配未命中的 → 标记，由 LLM 小模型做最终映射
- 抽检 3%，验证章节映射准确率

## 6. 阶段 2：实体抽取（LLM json_object）

唯一使用 LLM 的阶段之一。从 M&M 章节提取 13 类实体。

### 6.1 调度策略

三模块分批并发：
- **模块 C（前置）**：替代物实体 —— 模块 A/B 依赖
- **模块 A**：试验设计实体（Swine_Model / Swine / Intervention / Control_Group）
- **模块 B**：指标体系实体（Tissue_Site / Indicator / Method）

批次大小 50 篇/批，批内并发 10-15。

### 6.2 模块 C：替代物识别与对齐

**目标**：从 M&M 中识别所有干预物质，对齐 ALTERNATIVE.tsv 分类体系。

**前置：词表预处理**
- 建立 `name → (Alternative_Class, Subclass)` 索引
- 构建同义词映射表（如 `Lactobacillus plantarum` → `Lactiplantibacillus plantarum`）
- 生成模糊匹配索引（编辑距离 < 3）
- 生成正则变体模式（如 ZnO ← zinc oxide ← zinc oxide (ZnO)）

**LLM Prompt 核心指令**（嵌入 BACKGROUND.md 3.1 规则）：
- 严格禁止将多个实体字符串拼接为一个复合实体
- 复合制剂三步处理：拆分 → 逐个对齐 → 创建 Composite_Product 节点 + has_component 关系
- Other（单一未知物质）与 Composite_Product（多物质组合）严格互斥
- 保留原始英文全称及缩写到备注字段

**LLM 输出 JSON Schema**：
```json
{
  "doi": "string",
  "extraction_target": "alternatives",
  "alternatives": [{
    "entity_type": "Alternative",
    "standard_name": "string (词表标准名或原文名)",
    "abbreviation": "string | null",
    "alternative_class": "Plant_Extract | Trace_Element | Organic_Acid | Probiotic | Polysaccharides_and_Oligosaccharides | Enzyme | Bioactive_Peptides | Other",
    "subclass": "string | null",
    "match_source": "词表精确匹配 | 词表同义映射 | 词表模糊匹配 | Other_未匹配",
    "original_text": "string (原文中的原始表述)",
    "evidence_text": "string (原文证据句1-3句)",
    "source_location": "string (Methods 2.x)"
  }],
  "composite_products": [{
    "entity_type": "Composite_Product",
    "product_name": "string",
    "manufacturer": "string | null",
    "is_commercial": true,
    "components": [{ "standard_name": "string", "entity_type": "Alternative | Composite_Product" }],
    "original_text": "string",
    "evidence_text": "string",
    "source_location": "string"
  }],
  "warnings": ["string"]
}
```

### 6.3 模块 A：试验设计实体

**目标**：从 M&M 提取 Swine_Model, Swine, Intervention, Control_Group。

**LLM 输出 JSON Schema**（嵌入 BACKGROUND.md 3.2-3.4 规则，包括单位标准化换算规则）：
```json
{
  "doi": "string",
  "experiment_id": "string (DOI + Exp序号)",
  "description": "string",
  "swine_model": {
    "model_type": "challenge | normal",
    "stressor_name": "string | null",
    "challenge_method": "oral_gavage | injection | other | null",
    "challenge_dose": "string | null",
    "challenge_timing": "string | null",
    "evidence_text": "string",
    "source_location": "string"
  },
  "swine": {
    "breed": "string", "sex": "boar | sow | barrow | male | female | mixed",
    "age": "string", "physiological_stage": "string",
    "initial_body_weight": "string", "sample_size": 0,
    "evidence_text": "string", "source_location": "string"
  },
  "interventions": [{
    "intervention_target": "string (指向 Alternative.standard_name 或 Composite_Product.product_name)",
    "dose_value": 0.0,
    "dose_unit_original": "string",
    "dose_unit_standard": "mg/kg_feed | mg/kg_BW | g/kg | ppm | other",
    "administration_route": "diet | drinking_water | oral_gavage | injection | topical | other",
    "duration": "string",
    "basal_diet": "string",
    "positive_control": "string | null",
    "evidence_text": "string", "source_location": "string"
  }],
  "control_groups": [{
    "group_name": "string",
    "group_type": "negative_control | positive_control | basal_control | sham",
    "description": "string",
    "evidence_text": "string", "source_location": "string"
  }]
}
```

**后处理**：剂量单位标准化换算（1 ppm = 1 mg/kg，1% = 10000 mg/kg），保留原始单位表述。

### 6.4 模块 B：指标体系实体

**目标**：从 M&M 按 BACKGROUND.md 3.5 六类指标提取 Indicator / Tissue_Site / Method。

**LLM 输出 JSON Schema**：
```json
{
  "doi": "string",
  "tissue_sites": [{
    "site_name": "string",
    "site_category": "content | mucosa | serum | tissue | feces",
    "evidence_text": "string", "source_location": "string"
  }],
  "indicators": [{
    "standard_name": "string", "abbreviation": "string",
    "unit": "string",
    "indicator_category": "macro_phenotype | microbiome | metabolome | molecular",
    "measurement_method": "string",
    "measured_in": "string (Tissue_Site.site_name)",
    "evidence_text": "string", "source_location": "string"
  }],
  "methods": [{
    "method_name": "string", "description": "string",
    "evidence_text": "string", "source_location": "string"
  }]
}
```

**后处理**：指标-组织-方法映射完整性检查；指标名称跨文献标准化。

### 6.5 后处理与实体汇总

- 模块 C/A/B 产出合并
- 跨文献实体去重：Alternative 名称统一，建立 `has_synonym` 关系
- 按 DOI 聚合 → 去重 → 写入 `entities/` 目录（13 类实体各一个 TSV）

### 质量闸门（人工抽检 10-15%）
- 重点：Alternative 对齐正确率、Composite_Product 拆分执行率、剂量单位换算正确率
- 通过标准：实体对齐准确率 ≥ 85%

## 7. 阶段 3：关系与结果抽取（LLM json_object）

唯一使用 LLM 的阶段之二。从 Results/Discussion 章节两遍处理：第一遍全量抽取，第二遍程序化对齐与校验。

### 7.1 Pass 1：全量结果抽取

**输入**（每篇文献的上下文）：
- Results + Discussion 章节全文
- 阶段 2 产出的 Indicator 列表（来自同一 DOI）
- 阶段 2 产出的 Control_Group 列表
- 阶段 2 产出的 Tissue_Site 列表

**抽取规则**（嵌入 BACKGROUND.md 第四节全部规则）：
- 全量抽取：所有报告了统计比较结果（不论 P 值）均需抽取
- 对比基准（4.1）：攻毒模型的结果必须 compared_to 模型攻毒组
- 关系类型（4.2）：生长/消化/形态/代谢 → increases/decreases；基因/蛋白 → upregulates/downregulates；微生物丰度 → enriches/depletes
- 按四层级（4.5）：宏观表型 → 微生态组学 → 代谢组学与生化 → 分子表达
- 证据句（4.3）：1-2 句核心原句，不可大段复制；同一句被多行引用则一对多复制
- Result 属性（4.4）：完整填写 direction, p_value, significance_level, effect_size, time_point, subgroup, evidence_text, source_location

**LLM 输出 JSON Schema**：
```json
{
  "doi": "string",
  "results": [{
    "indicator_abbreviation": "string",
    "tissue_site": "string",
    "direction": "increased | decreased | no_significant_change",
    "relation_type": "increases | decreases | upregulates | downregulates | enriches | depletes | affects",
    "p_value": 0.0, "p_value_original_text": "string",
    "corrected_significance": "string | null",
    "significance_level": "p_less_0.01 | p_less_0.05 | trend_0.05_0.1 | not_significant",
    "effect_size": "string | null",
    "time_point": "string | null",
    "subgroup": "string | null",
    "compared_to_group": "string (Control_Group.group_name)",
    "evidence_text": "string (1-2句核心原文)",
    "source_location": "string (Results 3.x / Table x / Figure x)"
  }]
}
```

### 7.2 Pass 2：指标对齐与关系构建（程序化）

1. **缩写匹配**：Result 的 `indicator_abbreviation` → 阶段 2 `Indicator.abbreviation`。精确匹配优先，模糊匹配标记人工
2. **部位匹配**：Result 的 `tissue_site` → 阶段 2 `Tissue_Site.site_name`
3. **Result 实体生成**：一条 Result 对应一个 Result 实体，写入 `entities/results.tsv`
4. **关系链构建**：
   - `Intervention -[increases/decreases/upregulates/...]-> Result`
   - `Result -[corresponds_to]-> Indicator`
   - `Result -[occurs_in]-> Tissue_Site`
   - `Result -[compared_to]-> Control_Group`

### 7.3 质量校验（程序化）

- **对比基准校验**：攻毒模型下 Result.compared_to 必须指向模型对照组 → 违规标记
- **关系-方向一致性**：relation_type = "increases" ↔ direction = "increased" → 不匹配标记
- **指标覆盖检查**：阶段 2 定义的 Indicator → 阶段 3 是否有对应 Result，未覆盖的标记"未报告显著结果"

### 质量闸门（人工抽检 15-20%）
- 重点：不显著结果是否被遗漏、compared_to 基准是否正确、关系词与 direction 是否一致
- 通过标准：结果完整性 ≥ 85%，关系方向准确率 ≥ 90%

## 8. 阶段 4：一致性验证与双轨导出（程序化）

### 8.1 Schema 级校验
- 必填字段非空检查
- 枚举值合法性校验（所有枚举字段）
- 外键完整性：Intervention.uses → Alternative 存在性；Result.corresponds_to → Indicator 存在性
- 实体-关系类型匹配

### 8.2 业务规则校验
- 对比基准规则（BACKGROUND 4.1）
- 关系方向一致性
- 复合制剂完整性：Composite_Product 必须有 ≥1 个 has_component；Other 不得有 has_component
- 单位标准化验证
- evidence_text 完整性

### 8.3 校验报告
按严重程度分级输出：
- **FATAL**：必须修正才能导出
- **WARNING**：建议人工复核
- **INFO**：统计信息

FATAL 全部清零 → 通过闸门 → 导出。

### 8.4 TSV 导出
- 13 个实体 TSV（每类一个文件），列名对应 SCHEMA.tsv 属性
- 6 类关系 TSV（按关系名分文件）
- 1 个总表 `all_entities.tsv`
- 编码 UTF-8-BOM

### 8.5 Neo4j 导出
- 自动生成 Cypher 脚本 `import.cypher`
- 节点用 `MERGE` 防重，以合成 ID（DOI + experiment_id + 实体序号）为唯一键
- 关系创建前 MATCH 两端节点
- 自动生成索引语句
- 自动生成统计查询（节点数/关系数/按分类分布）

## 9. 技术实现要点

### 9.1 Python 依赖

```
anthropic / openai      # LLM API
lxml                    # XML 解析
jsonschema              # JSON Schema 校验
pandas                  # TSV 读写与数据处理
asyncio / httpx         # 异步并发
biopython               # PubMed E-utilities 封装
```

### 9.2 LLM 调用规范

- 所有 LLM 调用使用 `response_format: { type: "json_object" }`，绑定目标 JSON Schema
- 解析失败 → 标记 re-extract，最多重试 2 次
- 每个 LLM 输出文件携带 `_model`, `_timestamp`, `_retry_count` 元信息

### 9.3 文件组织

```
zhongnong/
├── data/
│   ├── xml/                    # 原始 XML 文件
│   ├── literature_pool.tsv     # 阶段0产出
│   ├── structured_sections/    # 阶段1产出 (逐文献 JSON)
│   ├── entities/               # 阶段2产出 (13类 TSV)
│   ├── relationships/          # 阶段3产出 (6类 TSV)
│   └── output/
│       ├── tsv/                # 最终 TSV 包
│       └── neo4j/              # import.cypher
├── src/
│   ├── stage0_search.py        # PubMed 检索 + 排重
│   ├── stage1_xml_parser.py    # XML 解析 + 章节结构化
│   ├── stage2_entity_extract.py    # 实体抽取调度器
│   ├── stage2_prompt_C.py      # 模块C Prompt模板
│   ├── stage2_prompt_A.py      # 模块A Prompt模板
│   ├── stage2_prompt_B.py      # 模块B Prompt模板
│   ├── stage3_result_extract.py    # 结果抽取 + 关系构建
│   ├── stage4_validate.py      # 校验引擎
│   ├── stage4_export_tsv.py    # TSV 导出
│   └── stage4_export_neo4j.py  # Neo4j Cypher 导出
├── prompts/                    # Prompt 模板文件
├── schemas/                    # JSON Schema 定义
│   ├── alternatives.json
│   ├── experiment_design.json
│   ├── indicators.json
│   └── results.json
├── ALTERNATIVE.tsv
├── SCHEMA.tsv
├── BACKGROUND.md
└── docs/superpowers/specs/
    └── 2026-06-01-knowledge-graph-pipeline-design.md
```

## 10. 人工复核节点总览

| 节点 | 抽检比例 | 检查重点 | 通过标准 |
|------|---------|---------|---------|
| 阶段 1 后 | 3% | XML 章节映射准确率 | ≥ 95% |
| 阶段 2 后 | 10-15% | Alternative 对齐、Composite 拆分、剂量单位 | ≥ 85% |
| 阶段 3 后 | 15-20% | 结果完整性、compared_to 基准、关系方向 | ≥ 85%/90% |
| 阶段 4 后 | 5% | 最终数据质量终审 | FATAL = 0 |
