from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型路径
model_path = "/mnt/model/DianJin-OCR-R1/seal_sft"
# 图片路径
image_path = "data/images/test/3009.jpg"

instruction = "请识别图片中的印章抬头。"
tool1 = "IXMTD5JPXGG9FEG10N  发票专用章  湄潭县何彬私房菜店"
tool2 = "上海鸿路何彬私房菜连锁店 发票专用章"
tools = """<tool>
以下是其它工具对该印章的识别内容：
{{
    "ocr_tool_1": "{tool1}",
    "ocr_tool_2": "{tool2}" 
}}
</tool>
"""

llm = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 10, "video": 10},
    gpu_memory_utilization=0.4,
)
processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": instruction},
        ],
    },
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.05,
    max_tokens=4096,
    stop=["<tool>"],
)

llm_inputs = [
    {
        "prompt": prompt,
        "multi_modal_data": mm_data
    }
]

outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
think_content = outputs[0].outputs[0].text.strip()
print("#" * 20 + " think " + "#" * 20)
print(think_content)

llm_inputs[0]["prompt"] = (
    llm_inputs[0]["prompt"].strip()
    + "\n"
    + think_content
    + "\n"
    + tools.format(tool1=tool1, tool2=tool2)
)
sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, repetition_penalty=1.05, max_tokens=4096
)
outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
rethink_content = outputs[0].outputs[0].text.strip()
print("#" * 20 + " rethink " + "#" * 20)
print(rethink_content)
