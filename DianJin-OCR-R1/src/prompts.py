seal_prompt_1 = """你是一名专业的OCR专家，你的任务是识别图片中的印章抬头。\
印章抬头位于印章上方，通常用于标明印章的归属、用途或机构名称，是印章的重要组成部分，帮助识别印章的合法性和适用范围。\
请将你的识别结果放在<recognition></recognition>之间。
"""

seal_prompt_2 = """以下是其它工具对该印章的识别内容：
<tool>
{{
    "ocr_tool_1": "{tool1}",
    "ocr_tool_2": "{tool2}" 
}}
</tool>
请比较你初步识别的结果和其它工具识别内容之间的**全部**不同之处。当存在差异时，你需要重新查看图片，并分析你的初步识别结果是否错误或者是否有遗漏。（你跟其他 OCR 工具均有可能犯错。请务必再次仔细查看图片）
将你的对比与分析过程放在<rethink></rethink>之间，并将最终识别的**印章抬头**内容放在<answer></answer>之间。"""

table_prompt_1 = """请将图中的表格以 HTML 格式输出。"""

table_prompt_2 = """以下是其它工具识别的 HTML 格式的内容：
<tool>
## 工具1：
{tool1}
## 工具2：
{tool2}
</tool>
请重新仔细查看图片，比较之前识别的结果和其它工具结果，分析表格结构(如单元格是否和图中一致、合并单元格 colspan 和 rowspan 是否正确等)和表格内容(如单元格中的内容是否识别错误或有遗漏)。如果有合并单元格，请以 colspan="x"/rowspan="x" 表示。
将分析过程放在<rethink></rethink>之间，并将最终识别的 HTML 放在<answer></answer>之间。

## 输出格式：
<rethink>
...
</rethink>
<answer>
HTML
</answer>"""

formula_prompt_1 = """请用 LaTeX 格式写出图中公式的表达式。"""

formula_prompt_2 = """以下是其它工具识别的 LaTeX 格式的内容：
<tool>
## 工具1结果：
{tool1}
## 工具2结果：
{tool2}
</tool>
请重新仔细查看图片，比较之前识别的结果和其它工具结果之间的不同之处，分析是否识别错误或有遗漏。
将分析过程放在 <rethink></rethink> 之间，并将最终识别的 LaTeX 放在 <answer></answer> 之间。

## 输出格式：
<rethink>
...
</rethink>
<answer>
LaTeX
</answer>"""