# 抗生素替代物知识图谱构建规范

本文档整合了实体类型、关系定义及详细的文献抽取规范，用于指导从科学文献中构建抗生素替代物领域的知识图谱。所有提取必须基于全文，并严格遵循以下规则。

---

## 1. 实体类型与属性（含抽取规范）

### 1.1 Alternative（抗生素替代物物质）

| 属性 | 类型 | 说明 | 抽取规范 |
|------|------|------|----------|
| standard_name | string | 标准名称 | 优先匹配《Alternative分类》表中的名称；若不在表中，保留原文名称，分类设为 Other |
| abbreviation | string | 常见缩写（仅一个最常用缩写，多个用英文分号分隔） | 提取原文中明确给出的缩写 |
| cas_number | string | CAS号（可选） | 若提供则填写，否则标注 “Not reported” |
| source_organism | string | 来源生物，如 *Thymus vulgaris* | 学名统一使用斜体 |
| is_synthetic | boolean | 是否合成 | TRUE / FALSE |
| evidence_text | text | 原文证据句子 | 描述该物质的原文原句（1-3句） |
| source_location | string | 来源位置 | 如 Table 1, Methods 2.3 |

**抽取规范**：
- **实体对齐**：优先使用提供的分类表进行映射，未涵盖的单一物质分类标记为 `Other`。
- **复合制剂**：严格按“拆分组分 → 创建 Composite_Product 节点 → 建立 has_component 关系”三步处理，**严禁**将组分名称拼接为长字符串作为Alternative节点名。
- **原始保留**：必须在额外字段（或备注）中保留原文中的原始英文全称及缩写。
- **抗生素排除**：所有用作阳性对照的抗生素（如金霉素、杆菌肽、粘菌素等）**严禁**提取为 `Alternative` 实体，其信息记录在对应阳性对照 `Intervention` 的属性中。

### 1.2 Alternative_Class（物质分类）

| 属性 | 类型 | 说明 |
|------|------|------|
| class_name | string | 分类名称（如 Plant_Extract, Probiotic, Other） |
| level | integer | 层级（1 或 2） |
| description | string | 分类定义 |

> **重要**：该实体为人工提前定义，不需要文本提取。参考 [抗生素替代物分类表格](./ALTERNATIVE.tsv)

### 1.3 Composite_Product（复合产品）

| 属性 | 类型 | 说明 | 抽取规范 |
|------|------|------|----------|
| product_name | string | 产品名称或唯一标识符 | 使用文献中给出的产品名（如 “Product X”） |
| manufacturer | string | 生产商 | 若有则填，否则 “Not reported” |
| is_commercial | boolean | 是否商业产品 | TRUE / FALSE |
| evidence_text | text | 原文证据 | 描述该复合产品的原文原句 |
| source_location | string | 来源位置 | |

> **重要**：`Composite_Product` 仅用于多物质组合，与 `Other`（单一物质但无法归类）互斥。

### 1.4 Literature（文献）

| 属性 | 类型 | 抽取规范 |
|------|------|----------|
| doi | string | 唯一标识符 |
| pmid | string | |
| title | string | |
| journal | string | |
| abstract_conclusion | text | **提取规则**：定位摘要最后1-3句，寻找结论性提示词（In conclusion, These results suggest, Overall, Therefore, In summary, Collectively）。若存在，完整复制该句；否则取最后一句，并标注“无明确结论提示词”。 |
| publication_year | integer | |
| publication_date | string | |
| study_design | enum | 完全随机、随机区组、析因、交叉等 |

> **重要**：Literature 实体的创建通过工程手段从PMC全文内容完成，不依赖LLM提取。

### 1.5 Experiment（试验）

| 属性 | 类型 | 说明 |
|------|------|------|
| experiment_id | string | 唯一标识（如 Literature_PMID + Exp_01） |
| description | text | 试验简述 |
| evidence_text | text | 原文证据 |
| source_location | string | |

### 1.6 Swine_Model（动物模型）

| 属性 | 类型 | 抽取规范 |
|------|------|----------|
| model_type | enum | `challenge` 或 `normal` |
| stressor_name | string | 应激源标准名称（如 E. coli K88, LPS, Diquat）。**若为复合应激**，可直接写组合名称（如“E. coli K88 + LPS”），并在 `evidence_text` 中保留原句。 |
| challenge_method | enum/string | oral_gavage, injection, other。**复合应激**可填写组合方式（如“oral_gavage + injection”） |
| challenge_dose | string | 如 10^9 CFU |
| challenge_timing | string | 如 day 7 |
| evidence_text | text | 描述模型构建的原句 |
| source_location | string | |

**抽取规范**：
- **可选实体**：若无有效信息，直接跳过该实体。
- **多应激源处理**：当动物接受多种应激时，可将多个应激合并描述在同一个 `Swine_Model` 中，也可创建多个 `Swine_Model` 实体并通过多次 `uses_model` 关联。推荐优先使用合并描述以减少节点冗余，示例：`stressor_name = "E. coli K88 + LPS"`, `challenge_method = "oral_gavage + injection"`。
- 若文献无攻毒/应激处理，统一标注为“常规健康模型（Normal conditions）”，`model_type = normal`，`stressor_name = "None"`。
- 明确区分绝对空白组与模型对照组（记录为 Control_Group 实体，类型为 `challenged_control` 等）。

### 1.7 Swine（动物参数）

| 属性 | 类型 | 抽取规范 |
|------|------|----------|
| breed | string | 品种（如 Duroc × Landrace × Yorkshire） |
| sex | enum | **优化枚举**：`boar`（公猪）, `sow`（母猪）, `barrow`（阉公猪）, `gilt`（后备母猪）, `male`（仅提及雄性未特指）, `female`（仅提及雌性未特指）, `mixed`（混合性别）。优先使用精确术语。 |
| age | string | 初始日龄或生理阶段（如 21 days, weaned） |
| physiological_stage | string | 如 weaned, grower, finisher |
| initial_body_weight | string | 含数值和单位（如 7.5 ± 0.2 kg）；若为范围则注明（如 “5.2 – 6.8 kg”） |
| sample_size | integer | 每组动物数 |
| evidence_text | text | 原文描述原句 |
| source_location | string | |

### 1.8 Intervention（干预措施）

| 属性 | 类型 | 抽取规范与单位换算 |
|------|------|-------------------|
| group_label | string | **处理组标签**，文献中该组的原始代号（如“T1”“500 mg/kg OEO”），用于区分多组干预 |
| dose_value | float | 数值（换算后）。对于复合产品，此值为产品总添加量。 |
| dose_unit_original | string | 原始单位（如 mg/kg, ppm, %, CFU/kg, U/kg） |
| dose_unit_standard | string | **标准化单位**：常用 `mg/kg_feed`, `mg/kg_BW`, `CFU/kg_feed`, `U/kg_feed`；新增 `U/kg_feed` 用于酶制剂。无法归类时填 `other`，同时必须在 `dose_unit_original` 中保留原始单位 |
| administration_route | enum | diet, drinking_water, oral_gavage, injection, topical, other |
| duration | string | 如 28 days |
| basal_diet | string | 基础日粮类型（如 corn-soybean meal） |
| substance_name | string | **仅阳性对照（抗生素）填写**：记录抗生素的标准名称（如 Chlortetracycline），若为复合抗生素则写主要成分。替代物干预一律填 “Not applicable”。 |
| evidence_text | text | 描述给药方案的原句 |
| source_location | string | |

> **注意**：
> - 若存在多个剂量梯度，每个剂量组创建独立的 `Intervention` 实体。
> - **阳性对照**：阳性抗生素处理需单独创建 `Intervention` 实体，在 `substance_name` 中填写抗生素名称，并配合 `group_label`、`dose_value`、`administration_route` 等属性完整记录给药方案；**严禁**为此创建 `Alternative` 实体或使用 `uses` 关系。
> - **物质关联**：干预与物质（单一或复合）的关联仅通过 `uses` 关系表达。
> - 未报告的字段标记为 “Not reported”。
> - 若同一干预组同时应用于不同模型（如正常和攻毒），只需创建一个 `Intervention` 实体，并通过多个 `applied_to` 关系分别关联对应的 `Swine_Model`。
> - **物质关联**：替代物干预通过 `uses` 关系关联 `Alternative` 或 `Composite_Product`；阳性抗生素对照**不使用 `uses` 关系**，其物质名称直接记录在 `substance_name` 属性中。

### 1.9 Control_Group（对照组）

| 属性 | 类型 | 说明 |
|------|------|------|
| group_name | string | 如 Challenged Control |
| group_type | string | **枚举值**：`negative_control`（绝对空白，无任何处理且无应激）、`positive_control`（抗生素或已知有效物质对照）、`basal_control`（基础日粮对照，仅限营养试验）、`sham`（假处理，如注射生理盐水）、`challenged_control`（攻毒/应激模型对照，给予应激但无替代物干预） |
| description | string | 详细描述 |
| evidence_text | text | |
| source_location | string | |

> **重要**：攻毒模型中，干预效果的对比基准**必须**是 `challenged_control`，而非绝对空白组。对照组类型必须根据其在实验中的角色准确选择。

### 1.10 Tissue_Site（组织部位）

| 属性 | 类型 | 说明 |
|------|------|------|
| site_name | string | 如 cecal content, jejunal mucosa, serum, Whole body |
| site_category | enum | content, mucosa, serum, tissue, feces, whole_organism |
| evidence_text | text | 原文证据句 |
| source_location | string | |

**补充说明**：宏观生产性能指标（如 ADG、死亡率、饲料转化率），其取样部位统一定义为预定义节点 `site_name="Whole body"`，`site_category="whole_organism"`。其他常见部位应尽量使用标准名称（参考附录术语表），以保持一致性。

### 1.11 Indicator（指标）

| 属性 | 类型 | 抽取规范 |
|------|------|----------|
| standard_name | string | 标准全称（如 Average Daily Gain） |
| abbreviation | string | 缩写（如 ADG） |
| unit | string | 单位（如 g/d, μm, μmol/g） |
| indicator_category | enum | macro_phenotype, microbiome, metabolome, molecular |
| evidence_text | text | **必须包含**描述该指标测定方法、取样部位、操作步骤的原文原句（1-3句） |
| source_location | string | |

**指标映射规范**：每个指标必须明确绑定：
- **取样部位**（通过 `measured_in` 关系关联 `Tissue_Site`，宏观指标统一关联到 `Whole body`）
- **测定方法**（通过 `uses_method` 关系关联 `Method`）

**常见指标分类示例**：
- **生长性能**：ADG, ADFI, F:G, FBW (注明阶段和单位)
- **消化率**：严格区分全肠道表观、回肠表观(AID)、回肠标准(SID)，并指明营养素（DM, CP, GE等）
- **肠道形态**：明确肠段（duodenum, jejunum, ileum）及指标（VH, CD, VH/CD）
- **微生物组**：取样部位（cecal content, feces等），α多样性（Shannon, Chao1），β多样性（PCoA/NMDS），门/属/种相对丰度
- **代谢组**：SCFAs（乙酸、丙酸、丁酸等），胆汁酸，生物胺等（注明浓度单位）
- **基因/蛋白表达**：组织（如 jejunal mucosa），靶标（ZO-1, Claudin-1, TNF-α等），内参，方法（qPCR, Western blot）
- **血清生化**：IgA, IgG, LPS, SOD, MDA, GLU, BUN等

### 1.12 Method（测定方法）

| 属性 | 类型 | 说明 |
|------|------|------|
| method_name | string | 如 qPCR, ELISA, weighing |
| description | string | 详细描述，可包含仪器、试剂盒等 |
| evidence_text | text | |
| source_location | string | |

### 1.13 Result（结果）

| 属性 | 类型 | 抽取规范 |
|------|------|----------|
| p_value | float | 具体数值（如 0.023），若无则留空 |
| p_value_original_text | string | 原文 _P_ 值表述（如 _P_ < 0.05, _P_ = 0.023, not significant）。若报道校正后 _P_ 值，也记录于此。 |
| corrected_significance | string | 多重比较校正方法及阈值（如 “FDR < 0.05”, “Tukey-adjusted _P_ < 0.05”）。无校正则填 “Not reported”。 |
| effect_size | string | 如 “fold change 1.5”, “mean ± SD: 3.2 ± 0.4 vs 2.1 ± 0.3” |
| time_point | string | 测定时间点（如 day 14） |
| subgroup | string | 亚组（如 barrow），若未分亚组填 “Not reported” |
| significance_level | enum | **基于最终结论（考虑校正）**：`p_less_0.01`, `p_less_0.05`, `trend_0.05_0.1`, `not_significant` |
| evidence_text | text | **精准提取**：描述该结果的完整英文原句（1-2句），若一句话含多个结果，分别复制到每个Result实体 |
| source_location | string | 如 Table 2, Figure 3, Results 2.1 |

**抽取总则**：所有报告了统计学比较的结果（无论 _P_ 值大小）均需抽取。

**方向表达规则**：结果的变化方向完全通过 `Intervention` 与 `Result` 之间的关系类型表达（`increases/decreases`, `upregulates/downregulates`, `enriches/depletes`, `affects`）。无显著差异的结果统一使用 `affects` 关系，并设置 `significance_level = not_significant`。

**对比基准规则**：通过关系 `compared_to` 关联 `Control_Group`。攻毒模型中，干预效果的对比基准**必须**是 `challenged_control`。此版本**暂不抽取干预间比较**（如剂量1 vs 剂量2），仅保留与对照组的比较。

**结果三元组**：每个 `Result` 实体隐含（部位 + 指标 + 变化方向），并通过 `corresponds_to` 关联 `Indicator`，通过 `occurs_in` 关联 `Tissue_Site`。

---

## 2. 关系类型及使用规范

### 2.1 分类与归属

| 关系 | 头实体 | 尾实体 | 含义 | 使用规范 |
|------|--------|--------|------|----------|
| belongs_to | Alternative | Alternative_Class | 物质分类 | |
| has_component | Composite_Product | Alternative / Composite_Product | 复合产品组分 | 支持嵌套复合；组分剂量等细节请填充在 Composite_Product.component_details |

### 2.2 文献与试验设计

| 关系 | 头实体 | 尾实体 | 含义 |
|------|--------|--------|------|
| contains | Literature | Experiment | |
| uses_model | Experiment | Swine_Model | 一次实验可关联多个 Swine_Model（如复合应激时多个单应激模型） |
| uses_animal | Experiment | Swine | |
| has_intervention | Experiment | Intervention | |
| measures_indicator | Experiment | Indicator | 记录本实验检测的所有指标 |
| uses_control | Experiment | Control_Group | |
| applied_to | Intervention | Swine_Model | 干预施加于哪些动物模型，支持一对多 |
| uses | Intervention | Alternative / Composite_Product | 干预使用的替代物物质（单一或复合） | **仅用于替代物**，不得用于阳性抗生素对照 |

> **说明**：若同一干预组用于多个模型（如正常与攻毒），通过多个 `applied_to` 连接实现。

### 2.3 指标与组织/方法

| 关系 | 头实体 | 尾实体 | 含义 |
|------|--------|--------|------|
| measured_in | Indicator | Tissue_Site | 指标测定的组织部位（宏观指标指向 `Whole body`） |
| uses_method | Indicator | Method | 测定方法（若使用多种方法，建立多条关系） |

### 2.4 结果关系（选择规范）

| 关系 | 头实体 | 尾实体 | 适用指标类型 | 使用规范 |
|------|--------|--------|-------------|----------|
| increases / decreases | Intervention | Result | 连续数值型表型：生长性能、消化率、代谢物浓度、血清生化、腹泻率（decreases表示改善）、死亡率等 | 根据变化方向二选一 |
| upregulates / downregulates | Intervention | Result | 基因或蛋白表达量（mRNA、蛋白丰度） | 针对表达水平的变化 |
| enriches / depletes | Intervention | Result | 微生物相对丰度（门/属/种） | 区分富集与减少 |
| affects | Intervention | Result | **兜底关系**：无法明确归类时使用；无显著变化的结果统一使用此关系 | 所有 `not_significant` 结果必须使用 `affects` |

| 其他结果关系 | 头实体 | 尾实体 | 含义 |
|------|--------|--------|------|
| corresponds_to | Result | Indicator | 结果对应哪个指标 |
| occurs_in | Result | Tissue_Site | 结果发生的部位 |
| compared_to | Result | Control_Group | 对比基准（**仅抽取与对照组的比较**，干预间比较暂不收录） |

> **方向一致性约束**：每个 `Result` 必须通过上述方向关系之一与 `Intervention` 相连，方向信息不得重复记录在 `Result` 属性中。

### 2.5 其他关系

| 关系 | 头实体 | 尾实体 | 含义 |
|------|--------|--------|------|
| has_synonym | Alternative | Alternative | 同义关系（如同一物质的不同命名） |
| leads_to | Indicator | Indicator | 因果通路（如某菌属丰度 → SCFA 浓度） |
| correlates_with | Indicator | Indicator | 统计相关 |
| part_of | Indicator | Indicator | 层级从属（如丁酸 → SCFAs） |

---

## 3. 通用抽取规则

### 3.1 证据溯源
- 每个从文献中直接抽取的实体（Alternative, Composite_Product, Experiment, Swine_Model, Swine, Intervention, Control_Group, Tissue_Site, Indicator, Result, Method）**必须**包含 `evidence_text` 和 `source_location`。
- `evidence_text` 应复制原文原句（1-3句），保持标点与用词不变。

### 3.2 材料与方法证据字段
- 对于每个 `Indicator`，其 `evidence_text` 必须包含描述该指标测定方法、取样部位、操作步骤的原文原句。若原文一段描述了多个指标，可将整段复制到每个指标记录中。

### 3.3 单位换算与标准化
- 剂量单位统一换算为标准化单位，如 `mg/kg_feed`、`mg/kg_BW` 等，同时保留原始单位在 `dose_unit_original` 中。
- 换算规则：1 ppm = 1 mg/kg；1% = 10000 mg/kg。
- **益生菌/酶制剂**：如原文为 CFU/kg 或 U/kg，可直接在 `dose_unit_standard` 中使用 `CFU/kg_feed` 或 `U/kg_feed`，无需强制转换为 mg。
- 所有无法归入上述类别的单位，`dose_unit_standard` 填写 `other`，并在 `dose_unit_original` 中完整保留原文单位及表达形式。

### 3.4 阴性结果处理
- 无显著差异的结果：使用 `affects` 关系连接 `Intervention` 和 `Result`，`significance_level` = `not_significant`，仍需记录 _P_ 值表述及效应量（如有）。

### 3.5 多剂量与多时间点
- 每个剂量梯度、每个时间点分别创建独立的 `Intervention` 或 `Result` 实体。

### 3.6 缺失数据标注
- 任何属性如果在全文中确实未提及，实体属性值统一填写字符串 `"Not reported"`（大小写如示例），严禁留空或使用 `N/A`。可选属性（如 CAS 号）若未提供可留空，但建议同样标注 `Not reported` 以确保处理一致。

---

## 4. 按四个层级的提取要点

| 层级 | 核心内容 | 关系词 | 注意事项 |
|------|----------|--------|----------|
| 宏观表型层 | ADG, ADFI, F:G, 消化率（明确类型+营养素），腹泻率，死亡率，VH/CD（明确肠段） | increases / decreases | 腹泻率降低用 decreases。所有宏观指标取样部位关联 `Whole body` |
| 微生态组学层 | α多样性，β多样性（分离/聚集），门/属/种丰度 | enriches / depletes | 必须绑定取样部位（盲肠内容物 vs 粪便） |
| 代谢组学与生化层 | SCFAs（乙酸/丙酸/丁酸），免疫球蛋白，炎症因子，抗氧化指标 | increases / decreases | 注明浓度单位。关系类型按数值型指标处理 |
| 分子表达层 | 紧密连接蛋白（ZO-1, Claudin-1），黏蛋白（MUC2），促炎/抗炎因子（TNF-α, IL-10），转运体（GLUT2, PEPT1） | upregulates / downregulates | 明确组织（空肠黏膜、肝脏等）及内参 |

---

*本规范适用于所有文献的全文本深度抽取，确保图谱构建的完整性、一致性与可溯源性。*