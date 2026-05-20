# 分类规则与Prompt工程参考

## 架构概览

本Skill采用 **前置解析 + 按需复用** 架构，核心流程：

```
材料图片 → analyze_materials.py (单次多图推理) → analysis_result.json
                                                    ↓ (--analysis-result)
            ┌──────────────────┬──────────────────┬────────────────────┐
            ↓                  ↓                  ↓                    ↓
   classify_claims.py   check_completeness   check_consistency   cross_validate
   (索引归档,0 Token)   (提取,0 Token)       (提取,0 Token)      (仅次模型)
```

### 设计原则

1. **一次推理，多次复用** - 前置解析做一次全量多图推理，提取分类、完整性、一致性所有信息
2. **按需编排** - 根据用户意图动态选择执行哪些下游步骤
3. **向后兼容** - 所有脚本保持独立运行能力，不传 `--analysis-result` 时自行调模型
4. **索引映射** - 分类归档使用文件系统实际索引，不依赖模型推测的文件名

---

## 前置解析 Prompt (analyze_materials.py)

前置解析采用 AICompete 风格综合型 Prompt，一次性从所有材料中提取全量信息：

```
# Role
你是一名资深保险理赔审核专家，同时具备 OCR 文字识别、医学知识和法律判断能力。

# Task
对上传的所有理赔材料图片进行综合分析，完成以下全部任务：

1. 逐图文档识别与分类
2. 关键信息提取（金额/日期/姓名/诊断等）
3. 材料完整性评估（是否缺件、是否重复）
4. 跨图交叉验证（身份一致性、时间线逻辑、费用匹配）
5. 风险预警标注

# Output Format (JSON)
{
    "case_summary": { "patient_name", "id_number", "claim_type", "total_amount", ... },
    "documents": [
        { "index", "filename", "document_type", "category_code", "sub_type", "key_info": {...} }
    ],
    "completeness_check": { "is_complete", "missing_materials", "present_categories" },
    "cross_validation": {
        "name_consistent", "timeline_consistent", "amount_consistent",
        "timeline_details", "financial_details"
    },
    "duplicate_materials": [...],
    "risk_warnings": [{ "type", "severity", "description", "related_files" }]
}
```

### 关键输出字段说明

| 字段 | 用途 | 下游消费者 |
|------|------|-----------|
| `documents[].category_code` | 6位分类编码(01-06) | classify_claims.py |
| `documents[].sub_type` | 细分类型 | classify_claims.py |
| `completeness_check` | 材料链完整性 | check_completeness.py |
| `cross_validation` | 跨图逻辑校验 | check_consistency.py |
| `risk_warnings` | 全类型风险点 | 所有下游脚本（按type过滤） |

---

## 大模型Prompt模板

调用多模态大模型时使用以下结构化Prompt进行图片内容识别与分类：

```
# Role
你是一名专业的保险理赔文档分析专家，擅长 OCR 文字识别与文档分类。

# Task
请分析上传的图片，完成以下任务：
1. **文档分类**：从以下列表中选择最匹配的一项：
   - 医疗发票（含住院/门诊/电子发票/其他发票）
   - 鉴定报告（含伤残鉴定/司法鉴定/资质证明/伤残会审表等）
   - 鉴定费用（鉴定费发票）
   - 病历资料（含入院记录/出院记录/诊断证明/疾病证明）
   - 费用清单（含住院清单/用药清单/医保结算单）
   - 其他材料（非上述材料分类，如辅助说明类材料）
2. **关键信息提取**：提取对理赔审核至关重要的文字内容（如：金额、日期、姓名、诊断结果、发票代码等）。

# Constraints
- 必须且仅输出合法的 JSON 格式，不要包含 markdown 标记（如 ```json）。
- 如果图片模糊无法识别，document_type 返回 "无法识别"，key_text 说明原因。
- 语言统一为简体中文。

# Output Format
{
    "document_type": "文档类型",
    "key_text": "关键文字内容摘要（请包含：总金额、开票日期、患者姓名、医疗机构名称等核心字段）"
}
```

## 分类优先级逻辑

分类采用三层判定机制：

### 第一层：LLM JSON解析

解析大模型返回的JSON中 `document_type` 字段，与预定义类型映射表匹配：

```python
type_mapping = {
    "住院发票": ("01_医疗发票", "住院发票"),
    "门诊发票": ("01_医疗发票", "门诊发票"),
    "电子发票": ("01_医疗发票", "电子发票"),
    "医疗发票": ("01_医疗发票", "医疗发票"),
    "收费票据": ("01_医疗发票", "医疗收费票据"),
    "伤残鉴定": ("02_鉴定报告", "伤残鉴定书"),
    "司法鉴定": ("02_鉴定报告", "司法鉴定意见书"),
    "司法鉴定意见书": ("02_鉴定报告", "司法鉴定意见书"),
    "鉴定报告": ("02_鉴定报告", "鉴定报告"),
    "鉴定费发票": ("03_鉴定费用", "鉴定费发票"),
    "鉴定费用": ("03_鉴定费用", "鉴定费发票"),
    "入院记录": ("04_病历资料", "入院记录"),
    "出院记录": ("04_病历资料", "出院记录"),
    "诊断报告": ("04_病历资料", "诊断报告"),
    "疾病证明": ("04_病历资料", "疾病证明书"),
    "疾病证明书": ("04_病历资料", "疾病证明书"),
    "病历资料": ("04_病历资料", "病历资料"),
    "住院费用清单": ("05_费用清单", "住院费用清单"),
    "费用清单": ("05_费用清单", "住院费用清单"),
    "用药清单": ("05_费用清单", "用药清单"),
    "医保结算单": ("05_费用清单", "医保结算单"),
}
```

### 第二层：病历资料细分

当文档属于"病历资料"大类时，基于 `key_text` 字段的关键词进一步细分：

| 细分类型 | 关键词正则 |
|---------|-----------|
| 出院记录 | `出院记录\|出院诊断\|出院医嘱\|出院小结\|出院情况` |
| 入院记录 | `入院记录\|入院诊断\|入院情况\|入院时间\|主诉.*入院` |
| 疾病证明书 | `疾病证明\|诊断证明\|病情证明` |
| CT影像报告 | `CT.*报告\|CT.*检查\|CT.*影像\|CT.*胶片` |
| MRI报告 | `MRI\|磁共振` |
| X线报告 | `X线\|X光\|DR报告` |
| 影像资料 | `影像\|胶片` |
| 检查报告 | `检查报告\|检验报告\|化验` |
| 诊断报告 | `诊断报告` |
| 手术记录 | `手术记录\|手术` |
| 病程记录 | `病程记录\|病程` |

### 第三层：正则规则兜底

当LLM响应无法解析或未匹配时，使用 `CATEGORY_RULES` 规则表按优先级逐条匹配：

```python
CATEGORY_RULES = [
    ("02_鉴定报告", "司法鉴定意见书", ["司法鉴定意见书", "法医学鉴定", "伤残等级鉴定"]),
    ("02_鉴定报告", "伤残鉴定书", ["伤残鉴定", "鉴定意见", "伤残等级", "劳动能力鉴定"]),
    ("03_鉴定费用", "鉴定费发票", ["鉴定费", "司法鉴定.*发票", "鉴定.*收费"]),
    ("04_病历资料", "入院记录", ["入院记录", "入院诊断", "入院情况", "入院时间"]),
    ("04_病历资料", "出院记录", ["出院记录", "出院诊断", "出院医嘱", "出院小结"]),
    ("04_病历资料", "疾病证明书", ["疾病证明", "诊断证明", "病情证明"]),
    ("04_病历资料", "诊断报告", ["诊断报告", "检查报告", "影像报告", "CT报告", "MRI", "X线"]),
    ("05_费用清单", "住院费用清单", ["费用清单", "住院费用", "费用明细", "收费明细"]),
    ("05_费用清单", "用药清单", ["用药清单", "药品清单", "处方笺"]),
    ("05_费用清单", "医保结算单", ["医保结算", "医疗保险结算", "社保结算", "基本医疗保险"]),
    ("01_医疗发票", "住院发票", ["住院.*发票", "住院收费", "住院票据"]),
    ("01_医疗发票", "门诊发票", ["门诊.*发票", "门诊收费", "门诊票据"]),
    ("01_医疗发票", "电子发票", ["电子发票", "增值税.*发票", "发票代码", "发票号码"]),
    ("01_医疗发票", "医疗收费票据", ["收费票据", "医疗收费", "门诊收费票据", "住院收费票据"]),
]
```

### 最终兜底：通用模式

如所有规则均未命中，使用宽泛关键词：
- 含"发票/票据/收费" → 01_医疗发票
- 含"鉴定/评定" → 02_鉴定报告
- 含"病历/诊断/医院/科室" → 04_病历资料
- 含"清单/明细/费用" → 05_费用清单
- 其余 → 06_其他材料

## API调用配置

### DashScope OpenAI兼容接口（单图/多图）

```python
from openai import OpenAI

client = OpenAI(
    api_key="<DASHSCOPE_API_KEY>",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 多图调用模式（用于 analyze_materials.py）
content = []
for file in files:
    content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
content.append({"type": "text", "text": "<SYSTEM_PROMPT>"})

completion = client.chat.completions.create(
    model="qwen3.5-plus",  # 多图推荐
    messages=[{"role": "user", "content": content}],
    max_tokens=8000
)
```

### DashScope SDK（Kimi-K2.5 调用）

```python
import dashscope
from dashscope import MultiModalConversation

dashscope.api_key = "<DASHSCOPE_API_KEY>"

# 多图 base64 输入
content = []
for file in files:
    content.append({"image": f"data:{mime};base64,{b64}"})
content.append({"text": "<SYSTEM_PROMPT>"})

response = MultiModalConversation.call(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": content}]
)
```

### 支持的图片格式

| 扩展名 | MIME类型 |
|--------|---------|
| .jpg/.jpeg | image/jpeg |
| .png | image/png |
| .gif | image/gif |
| .webp | image/webp |

## 性能参考数据

### 前置解析+复用模式（推荐）

基于22份理赔材料的全流程实测（2026-05-05）：

| 步骤 | 耗时 | 输入 Token | 输出 Token | 备注 |
|------|------|-----------|-----------|------|
| analyze_materials.py | 150.6s | 55,653 | 7,793 | qwen3.5-plus, 多图单次 |
| classify_claims.py -a | 0.1s | 0 | 0 | 索引归档，22/22 正确分类 |
| check_completeness.py -a | <0.1s | 0 | 0 | JSON提取，is_complete=True |
| check_consistency.py -a | <0.1s | 0 | 0 | JSON提取，全维度一致 |
| cross_validate.py -a | 134.8s | 88,196 | 3,946 | 仅Kimi次模型，confidence=1.0 |

全流程合计：~285s, ~155k Token

### 独立逐图模式（仅分类场景）

基于77个文件的历史实测：

| 指标 | 数值 |
|------|------|
| 平均每文件耗时 | ~8.7秒 |
| 平均Token消耗/文件 | ~3,292 |
| 总Token(77文件) | ~253,566 |
| 建议并发 | 串行处理（避免API限流） |

### 成本对比

| 模式 | 22文件 Token | 适用场景 |
|------|-------------|---------|
| 前置解析+全量复用 | ~155k | 多项检查组合（推荐） |
| 仅分类（逐图） | ~72k | 单独分类，无需其他检查 |
| 全部独立调用 | ~450k+ | 不推荐 |

---

## 文件名映射机制

### 问题背景

大模型在分析多张图片时，推理出的文件名可能与实际文件系统名称不一致。典型情况：模型在中文字符和数字之间添加空格（如 "鉴定意见1.jpg" → "鉴定意见 1.jpg"）。

### 解决方案：索引映射

```python
# 加载目录时记录实际文件名列表
file_list = get_file_info_list(input_dir)  # [{"file_name": "鉴定意见1.jpg", ...}, ...]

# 分类归档时使用索引对应
for idx, doc in enumerate(analysis["documents"]):
    # 优先使用 file_list 中的实际文件名（索引对应）
    filename = ""
    if idx < len(file_list):
        filename = file_list[idx].get("file_name", "")
    # 回退：使用排序后的源文件列表
    if not filename and idx < len(src_files):
        filename = src_files[idx]
    # 永不使用: doc.get("filename") ← 可能有空格等差异
```

### 保障措施

- `analyze_materials.py` 的 prompt 要求模型按输入顺序（index: 0, 1, 2...）返回 documents
- `utils.get_file_info_list()` 按相同排序规则生成文件列表
- 两者通过 index 对齐，消除文件名不匹配问题
- 实测22文件场景，归档成功率 100%

---

## 完整性检查 Prompt 设计 (check_completeness.py)

完整性检查使用单次多图调用，一次性将所有材料图片发送给模型，执行以下分析：

### 检查维度

1. **类型完整性** - 核心材料链是否缺失
   - 通用医疗：医疗发票 + 病历资料 + 费用清单
   - 有鉴定案件：+ 鉴定报告 + 鉴定费用发票
2. **页面完整性** - 多页文档是否缺页
3. **重复材料识别** - 发票/病历/清单等重复提交检测

### 重复识别规则

| 材料类型 | 判定方法 | 置信度 |
|---------|---------|--------|
| 发票 | 票据代码+号码+金额完全一致 | high |
| 发票 | 号码相同但打印时间不同 | medium |
| 病历/证明 | OCR文本相似度≥95% | high |
| 费用清单 | 住院号+合计+时间范围 | high |
| 同一发票不同形式(纸质/电子) | 识别为同一凭证 | high |

### 边界处理

- 同一住院多次结算 → 不判定为重复
- 连号发票/不同日期发票 → 不判定为重复
- 病历更新版本 → 不判定为重复
- 同一发票纸质版/电子版 → 判定为重复（建议保留清晰度最高的）

### 输出格式

JSON报告包含: `completeness_check` / `duplicate_materials` / `documents_summary` / `risk_warnings`

---

## 一致性检查 Prompt 设计 (check_consistency.py)

一致性检查专注于跨图逻辑关联，将所有材料作为一个整体进行校验。

### 检查维度

1. **身份一致性** - 所有图片中的患者姓名/身份证号是否一致
2. **时间线校验** - 入院日期 < 出院日期 ≤ 发票开具日期 < 鉴定日期
3. **费用逻辑** - 发票金额 vs 费用清单总额匹配（允许≤5%偏差）
4. **医疗逻辑** - 治疗项目与诊断是否相关

### 评分机制

一致性评分 (consistency_score): 0.0-1.0，其中：
- 1.0: 完全一致，无任何差异
- 0.8-0.99: 基本一致，存在轻微差异
- 0.6-0.79: 部分不一致，需关注
- <0.6: 存在重大不一致，需人工复核

### 输出格式

JSON报告包含: `consistency_result` / `identity_check` / `timeline_check` / `financial_check` / `medical_logic_check` / `risk_warnings`

---

## 交叉验证 Prompt 设计 (cross_validate.py)

交叉验证使用两个独立的大模型分别分析相同材料，然后程序化对比结果。

### 双模型架构

**独立运行模式：**
```
材料图片 → 主模型(Qwen3.5) → 分析结果A ─┐
                                           ├── 程序化对比 → 差异报告
材料图片 → 验证模型(Kimi-K2.5) → 分析结果B ─┘
```

**复用模式（推荐，传入 --analysis-result）：**
```
analysis_result.json → 直接作为结果A（0 Token）─┐
                                                  ├── 程序化对比 → 差异报告
材料图片 → 验证模型(Kimi-K2.5) → 分析结果B ──────┘
```

复用模式下节省主模型调用：~55k Token、~150s 耗时。

### 对比维度

1. **分类一致性** - 两个模型对每张图片的分类是否一致
2. **金额提取** - 两个模型提取的金额是否一致
3. **患者信息** - 两个模型提取的姓名/ID是否一致
4. **完整性判定** - 两个模型对材料完整性的判断是否一致
5. **时间线判定** - 两个模型对时间逻辑的判断是否一致

### 可信度评分

- 无差异: confidence = 1.0
- 每处差异降低 0.15，最低 0.4
- 建议阈值: confidence ≥ 0.7 可自动通过，< 0.7 建议人工复核

### Kimi 模型调用

Kimi-K2.5 通过 DashScope SDK (`dashscope.MultiModalConversation`) 调用：
- 支持多图 base64 输入
- 需要 `pip install dashscope`
- 使用与 Qwen 相同的系统提示词确保可比性
- JSON提取支持5级策略（直接解析→代码块→括号匹配→候选选择）

### 输出格式

JSON报告包含: `cross_validation_summary` / `primary_analysis` / `secondary_analysis`

---

## 共享工具模块 (utils.py)

所有检查脚本共享 `scripts/utils.py` 工具模块，提供：

- `LoadedFile` 数据结构 - 统一的文件加载结果
- `load_directory()` - 目录文件加载（支持图片+PDF）
- `resize_image()` - 图片等比例缩放（最大2048px）
- `image_to_base64()` - 图片转Base64编码
- `call_qwen_multimodal()` - Qwen模型调用（OpenAI兼容模式）
- `call_kimi_multimodal()` - Kimi模型调用（DashScope SDK）
- `extract_json_from_text()` - 5级JSON提取策略
- `save_json_report()` - JSON报告保存

---

## 扩展定制

### 添加新分类

在 `CATEGORY_RULES` 列表中添加新条目：
```python
("编号_分类名", "描述名称", ["关键词1", "关键词2"])
```

### 添加类型映射

在 `type_mapping` 字典中添加新的键值对：
```python
"新类型名": ("编号_分类名", "描述名称")
```

### 切换模型

修改 `--model` 参数即可。推荐模型配置：
- 前置解析（多图综合）: `qwen3.5-plus`（支持多图、推理强、综合最优）
- 分类（逐图处理）: `qwen-vl-max`（速度快、准确率高、单图场景）
- 交叉验证次模型: `kimi-k2.5`（独立模型体系，结果有差异性价值）

### 前置解析输出 Schema

`analysis_result.json` 的完整结构：

```json
{
    "case_summary": {
        "patient_name": "张三",
        "id_number": "3201...",
        "claim_type": "人身伤害",
        "hospital": "XX医院",
        "total_amount": 12345.67,
        "admission_date": "2024-01-01",
        "discharge_date": "2024-01-15"
    },
    "documents": [
        {
            "index": 0,
            "filename": "模型推测名(仅参考)",
            "document_type": "住院发票",
            "category_code": "01",
            "sub_type": "住院发票",
            "key_info": {
                "amount": "12345.67",
                "date": "2024-01-15",
                "patient_name": "张三",
                "hospital": "XX医院"
            }
        }
    ],
    "completeness_check": {
        "is_complete": true,
        "missing_materials": [],
        "present_categories": ["01_医疗发票", "02_鉴定报告", "04_病历资料", "05_费用清单"]
    },
    "cross_validation": {
        "name_consistent": true,
        "timeline_consistent": true,
        "amount_consistent": true,
        "timeline_details": "入院2024-01-01 < 出院2024-01-15 < 发票2024-01-15 ✓",
        "financial_details": "发票金额与清单合计一致"
    },
    "duplicate_materials": [],
    "risk_warnings": [
        {
            "type": "completeness|consistency|timeline|financial|medical_logic|identity|duplicate|page_missing",
            "severity": "high|medium|low",
            "description": "具体风险描述",
            "related_files": ["文件1.jpg", "文件2.jpg"]
        }
    ]
}
```

### 下游脚本复用规则

| 脚本 | 提取字段 | 过滤条件 |
|------|---------|---------|
| classify_claims.py | `documents[].category_code`, `documents[].sub_type` | - |
| check_completeness.py | `completeness_check`, `duplicate_materials`, `documents` | `risk_warnings` type in (completeness, duplicate, page_missing) |
| check_consistency.py | `cross_validation`, `case_summary` | `risk_warnings` type in (consistency, identity, timeline, financial, medical_logic) |
| cross_validate.py | 整体 JSON 作为主模型结果 | - |

### 自定义案件类型

完整性检查支持 `--claim-type` 参数指定案件类型，模型会据此调整必需材料列表：
- "通用医疗" - 默认，需要发票+病历+清单
- "交通事故" - 额外需要事故认定书
- "工伤" - 额外需要工伤认定决定书

---

## API-数据映射关系

本Skill的数据流转路径，确保每个数据的来源可追溯：

### 数据流入（输入）

| 数据项 | 来源 | 数据级别 | 获取方式 | 置信度 |
|--------|------|---------|---------|--------|
| 理赔材料图片 | 用户本地文件系统 | L3-敏感 | 目录扫描 | 100%(原始输入) |
| DashScope API Key | 用户手动输入 | L2-机密 | CLI参数 | - |
| 案件类型标注 | 用户手动输入 | L5-公开 | CLI参数 | - |

### 外部API调用

| API端点 | 调用方式 | 数据级别 | 传输内容 | 安全措施 |
|---------|---------|---------|---------|---------|
| `dashscope.aliyuncs.com/compatible-mode/v1` | HTTPS POST | L3-敏感 | Base64编码图片 + Prompt | TLS加密传输 |
| `dashscope.aliyuncs.com` (Kimi SDK) | HTTPS POST | L3-敏感 | Base64编码图片 + Prompt | TLS加密传输 |

### 数据流出（输出）

| 输出项 | 消费者 | 数据级别 | 存储位置 | 保留策略 |
|--------|--------|---------|---------|---------|
| analysis_result.json | 下游4脚本 | L3-敏感 | 本地文件 | 由用户管理 |
| 分类归档目录 | 理赔系统/人工 | L3-敏感 | 本地文件 | 由用户管理 |
| Markdown报告 | 理赔人员 | L4-内部 | 本地文件 | 由用户管理 |
| audit_log.jsonl | 合规审计 | L4-内部 | Skill目录 | 追加不可删除 |

### Skill间数据传递

```
analyze_materials.py
    │ 输出: analysis_result.json (L3)
    │
    ├──→ classify_claims.py
    │    读取: documents[].category_code, documents[].sub_type
    │    产出: 归档目录 + Markdown报告 (L3/L4)
    │
    ├──→ check_completeness.py
    │    读取: completeness_check, duplicate_materials
    │    产出: completeness_report.json (L4)
    │
    ├──→ check_consistency.py
    │    读取: cross_validation, case_summary, risk_warnings
    │    产出: consistency_report.json (L4)
    │
    └──→ cross_validate.py
         读取: 整体JSON作为主模型结果
         额外调用: Kimi API (L3传输)
         产出: cross_validation_report.json (L4)
```

---

## 审计日志规范

### 日志格式

审计日志采用 JSONL (JSON Lines) 格式，每次执行追加一条记录：

```json
{
    "audit_id": "uuid",
    "timestamp": "ISO-8601",
    "skill_version": "2.0.0",
    "input": {
        "file_count": 22,
        "input_dir_hash": "sha256前16位",
        "model": "qwen3.5-plus",
        "capabilities_requested": ["classify", "completeness"]
    },
    "execution": {
        "steps_executed": ["analyze", "classify", "check_completeness"],
        "total_tokens": 63446,
        "total_duration_seconds": 151.2,
        "errors": []
    },
    "output": {
        "output_path": "路径",
        "report_hash": "sha256前16位",
        "risk_flags": [],
        "confidence_scores": {"classification": 0.95, "completeness": 1.0}
    },
    "risk_disclosure": {
        "confidence": 0.95,
        "data_freshness": "实时处理",
        "disclaimer": "本审核结果由AI辅助生成..."
    }
}
```

### 日志存储

- 文件路径: `<skill_dir>/audit_log.jsonl`
- 写入模式: append-only (追加，不可修改历史)
- 保留策略: 长期保留，由合规部门定期归档
- 字段可追溯: 通过 `input_dir_hash` 和 `report_hash` 可重建完整审计链

---

## 置信度评估标准

各检查维度的置信度计算方式：

| 维度 | 计算方法 | 说明 |
|------|---------|------|
| 分类 | 模型确定性 + 正则规则命中 | 两者一致→0.95+；仅模型→0.85；仅正则→0.75 |
| 完整性 | 模型输出is_complete的确定性 | 模型+结构验证一致→0.95 |
| 一致性 | consistency_score (模型直接输出) | 模型综合评估的0-1评分 |
| 交叉验证 | 1.0 - 差异数 × 0.15 | 程序化计算，最低0.4 |

### 阈值定义

```yaml
thresholds:
  auto_pass: 0.9          # ≥0.9 可自动通过
  human_review: 0.7       # 0.7-0.9 建议人工抽检
  alert_trigger: 0.7      # <0.7 触发[ALERT]，暂停流程
  manual_required: 0.5    # <0.5 必须人工全面复核
```
