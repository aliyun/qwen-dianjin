import os
import json
from tqdm import tqdm
import requests
from mcp import ClientSession
from mcp.client.sse import sse_client
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import nest_asyncio
from typing import List, Tuple, Optional, Dict, Any
import random
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import yaml
import re
nest_asyncio.apply()

# ------- Inference Config ------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='infer_config.yaml', help='Path to the config file')
args = parser.parse_args()
if os.path.exists(args.config):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            print(key, value)
            setattr(args, key, value)
os.makedirs(os.path.dirname(args.answer_save_path), exist_ok=True)
# ------- Inference Config ------------


# ----- GLOBAL CONFIG -------
with open(args.MCP_SCHEMA_PATH, "r") as f:
    tools_list = json.load(f)

print(f"Loading model from {args.model_hf} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_hf)
model = AutoModelForCausalLM.from_pretrained(args.model_hf, device_map="auto")
# ----- GLOBAL CONFIG -------



class MCPClient:
    def __init__(self, max_retries: int = 3, retry_backoff: float = 1.0):
        self.session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff  # 初始退避时间（秒）

    async def connect_to_sse_server(self, server_url: str):
        for attempt in range(1, self.max_retries + 1):
            try:
                streams_context = sse_client(url=server_url)
                streams = await self._exit_stack.enter_async_context(streams_context)
                session_context = ClientSession(*streams)
                self.session = await self._exit_stack.enter_async_context(session_context)
                await self.session.initialize()
                return
            except Exception as e:
                print(f"[MCPClient] Connect failed (attempt {attempt}): {e}")
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed to connect after {self.max_retries} retries: {e}")
                backoff = self.retry_backoff * (2 ** (attempt - 1)) * (1 + random.random())
                print(f"[MCPClient] Retrying in {backoff:.1f}s...")
                await asyncio.sleep(backoff)

    async def cleanup(self):
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            print(f"[MCPClient] cleanup error: {e}")

    async def _call_tool_single(self, tool_name, tool_args, timeout: float = 60.0):
        for attempt in range(1, self.max_retries + 1):
            try:
                async with asyncio.timeout(timeout):
                    result = await self.session.call_tool(tool_name, tool_args)
                    return result.content[0].text
            except asyncio.TimeoutError:
                print(f"[MCPClient] Tool '{tool_name}' timeout on attempt {attempt}")
            except Exception as e:
                print(f"[MCPClient] Tool '{tool_name}' failed (attempt {attempt}): {e}")
     
            if attempt < self.max_retries:
                backoff = self.retry_backoff * (2 ** (attempt - 1)) * (1 + random.random())
                print(f"[MCPClient] Retrying '{tool_name}' in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
        raise RuntimeError(f"Tool '{tool_name}' failed after {self.max_retries} retries")

    
    async def parser_and_call_tools(self, query, timeout: float = 10.0):
        if isinstance(query, str):
            try:
                query = json.loads(query)
            except Exception:
                return {"response": "input format is error"}

        results = await asyncio.gather(*[
            self._call_tool_single(tool_call['name'], tool_call['arguments'], timeout)
            for tool_call in query
        ], return_exceptions=True)

        formatted = []
        for r in results:
            if isinstance(r, Exception):
                formatted.append(f"[error] {r}")
            else:
                formatted.append(r)
        return formatted
  

def exec_tool_batch(tool_list):
    """
    批量执行工具调用
    
    Args:
        tool_list: 工具列表，格式为 [{
            "name": tool_name,
            "arguments": arguments
        }, ...]
    
    Returns:
        list: 工具调用结果列表
    """
    
    # 每次调用创建新的客户端实例
    async def run_async_call():
        client = MCPClient(max_retries=args.mcp_max_retries)
        try:
            await client.connect_to_sse_server(args.MCP_SERVER_URL)
            # 添加超时机制
            result = await asyncio.wait_for(
                client.parser_and_call_tools(tool_list), 
                timeout=args.mcp_timeout
            )
            return result
        except asyncio.TimeoutError:
            print("工具调用超时")
            return None
        except Exception as e:
            print(f"异步调用错误: {e}")
            return None
        finally:
            try:
                await client.cleanup()
            except:
                pass
    
    try:
        # 使用现有的事件循环或创建新的事件循环
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 如果事件循环已经在运行，使用 asyncio.run_coroutine_threadsafe
        if loop.is_running():
            # 在线程中运行协程
            future = asyncio.run_coroutine_threadsafe(run_async_call(), loop)
            result = future.result(timeout=60)  # 设置超时时间
            return result
        else:
            result = loop.run_until_complete(run_async_call())
            return result
    except Exception as e:
        print(f"异步调用错误: {e}")
        import traceback
        traceback.print_exc()
        return None


class AssistantMessage:
    def __init__(self, reasoning_content, answer_content, tool_info):
        self.reasoning_content = reasoning_content
        self.answer_content = answer_content
        self.tool_info = tool_info

    def get_message_dict(self):
        return {
            "reasoning_content": self.reasoning_content,
            "content": self.answer_content,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool.get('id',index),
                    "function": {
                        "arguments": tool['arguments'] if isinstance(tool['arguments'], str) else json.dumps(tool['arguments']),
                        "name": tool['name']
                    },
                    "type": "function",
                    "index": index,
                } for index, tool in enumerate(self.tool_info)
            ] if len(self.tool_info) != 0 else None,
        }
        

def parse_llm_output(
    text: str,
    think_start: str = args.hf_think_start,
    think_end: str = args.hf_think_end,
    tool_start: str = args.hf_tool_start,
    tool_end: str = args.hf_tool_end,
):
    text = text.strip()
    if think_end in text and think_start not in text:
        text = think_start + text
    reasoning = []
    tool_calls = []
    answer_parts = []

    # 构造通用正则
    escaped_tags = map(re.escape, [think_start, think_end, tool_start, tool_end])
    t_start, t_end, tool_s, tool_e = escaped_tags
    pattern = re.compile(rf"({t_start}.*?{t_end}|{tool_s}.*?{tool_e})", re.S)

    last_end = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        before = text[last_end:start].strip()
        if before:
            answer_parts.append(before)

        block = match.group(0)
        if block.startswith(think_start):
            content = re.search(rf"{t_start}(.*?){t_end}", block, re.S).group(1).strip()
            reasoning.append(content)
        elif block.startswith(tool_start):
            content = re.search(rf"{tool_s}(.*?){tool_e}", block, re.S).group(1).strip()
            try:
                tool_calls.append(json.loads(content))
            except Exception:
                tool_calls.append(content)
        last_end = end

    if last_end < len(text):
        after = text[last_end:].strip()
        if after:
            answer_parts.append(after)
    
    
    return {
        "reasoning_content": "\n".join(reasoning).strip() or None,
        "tool_calls": tool_calls or [],
        "answer": "\n".join(answer_parts).strip() or None,
    }


def get_response_reasoning_with_tools(history_msgs,model,tools):
    text = tokenizer.apply_chat_template(
        history_msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
        tools=tools,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(268, content)
    print('===================\n\n')
    res_dict = parse_llm_output(content)
    reasoning_content, answer_content, tool_info = res_dict['reasoning_content'],res_dict['answer'],res_dict['tool_calls']
    return reasoning_content, answer_content, tool_info


# hf version
def call_with_messages(history_msgs,model):
    messages = history_msgs
    tools = tools_list
    reasoning_content, answer_content, tool_info = get_response_reasoning_with_tools(messages, model, tools)
    
    as_ = AssistantMessage(reasoning_content, answer_content, tool_info)
    messages.append(as_.get_message_dict())
    if len(tool_info) == 0:
        return messages
    tool_num = 0
    while len(tool_info) != 0 and tool_num <= args.max_tool_num:
        tool_num += len(tool_info)
        tool_calls = [{
            "name": tool['name'],
            "arguments": tool['arguments']
        } for tool in tool_info]
        batch_results = exec_tool_batch(tool_calls)
        # 处理结果
        if batch_results:
            tool_idx = 0
            for tool, result in zip(tool_info, batch_results):
                tool_info_message = {
                    "content": str(result) if result else "",
                    "role": "tool",
                    "tool_call_id": tool.get('id',tool_idx),
                }
                messages.append(tool_info_message)
                tool_idx += 1
        else:
            messages.append({
                    "content": "调用失败",
                    "role": "tool",
                })
        reasoning_content, answer_content, tool_info = get_response_reasoning_with_tools(messages, model, tools)
        as_ = AssistantMessage(reasoning_content, answer_content, tool_info)
        messages.append(as_.get_message_dict())
    return messages


def process_one_sample(idx: int, sample: Dict[str, Any],model) -> Tuple[int, Dict[str, Any], Dict[str, float]]:
    orig_traj: List[Dict[str, Any]] = sample.get("messages", [])    # query - gt     - query2  -  gt2   ... final answer.
    new_traj: List[Dict[str, Any]] = []                             # query - answer - query2  -  answer2(gt2) ... final answer. NOTE: answer2 depends on `gt` rather than `answer` 

    for i, turn in enumerate(orig_traj):
        role = turn.get("role")
        if role in ("system", "user"):
            new_traj.append(copy.deepcopy(turn))
        if role == "user":
            history_msgs = orig_traj[:(i+1)]
            msgs_QA = call_with_messages(history_msgs, model)
            try:
                assistant_turn = msgs_QA[(i+1):]
            except Exception as e:
                assistant_turn = [{"role": "ERROR", "content": "[Error] {}".format(str(e))}]
            new_traj.extend(assistant_turn)
    new_sample = dict(sample)
    new_sample["messages_pred"] = new_traj
    return idx, new_sample


def main_HF_inference():
    print(f"Loading data from {args.query_data_path}")
    with open(args.query_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[0:2] + data[200:202] 

    all_results_inst = []
    for idx, item in tqdm(enumerate(data), total=len(data), desc="HF Inference"):
        try:
            idx, new_sample = process_one_sample(idx, item, model)
            all_results_inst.append(new_sample)
        except Exception as e:
            print(f"[Error] idx={idx}: {e}")
            all_results_inst.append({"error": str(e), "messages_pred": []})

    with open(args.answer_save_path, "w", encoding="utf-8") as f:
        json.dump(all_results_inst, f, indent=4, ensure_ascii=False)
    print(f'{args.query_data_path}\nInference Done. Saved in {args.answer_save_path}')


if __name__ == "__main__":
    main_HF_inference()
