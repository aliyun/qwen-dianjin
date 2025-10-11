import transformers
import os
import copy
import json
import random
import torch
import logging
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer
from datasets import Dataset
from typing import Sequence, Dict
from multiprocessing import Pool
import os
import argparse
import yaml

# ------- Inference Config ------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='sft_config.yaml', help='Path to the config file')
args = parser.parse_args()
if os.path.exists(args.config):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            print(key, value)
            setattr(args, key, value)
# ------- Inference Config ------------

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

with open(args.MCP_SCHEMA_PATH,'r') as f:
    tools = json.load(f)

class TokenizerForMultiturns():

    def __init__(self, messages, tokenizer):
        self.input_ids = []
        self.labels = []
        self.test_message_list = messages
        self.curr_messages_list = []
        self.tokenizer = tokenizer

    def process_messages(self, messages_list, mask=False):
        """处理消息并更新input_ids和labels"""
        temp_input_ids = self.tokenizer.apply_chat_template(messages_list, tokenize=True, tools=tools)
        new_tokens = temp_input_ids[len(self.input_ids):]
        self.input_ids += new_tokens

        if mask:
            self.labels += [-100] * len(new_tokens)
        else:
            self.labels += new_tokens

    def get_tokenization_res(self):

        for index, message in enumerate(self.test_message_list):
            if index == 0:
                self.curr_messages_list.append(message)
                self.process_messages(self.curr_messages_list, mask=True)  # 第一条消息mask掉
                continue
            self.curr_messages_list.append(message)
            
            if message['role'] == 'assistant':
                # 处理assistant消息，不mask
                self.process_messages(self.curr_messages_list, mask=False)
                
            elif message['role'] == 'tool':
                # 检查是否是连续tool消息的最后一个
                if (index + 1 >= len(self.test_message_list) or 
                    self.test_message_list[index + 1]['role'] != 'tool'):
                    # 处理tool消息，需要mask
                    self.process_messages(self.curr_messages_list, mask=True)
        return self.input_ids, self.labels
    


if isinstance(args.raw_train_json_path, str):
    with open(args.raw_train_json_path, "r") as f:
        data = json.load(f)
elif isinstance(args.raw_train_json_path, list):
    data = []
    for data_path in args.raw_train_json_path:
        with open(data_path, "r") as f:
            data.extend( json.load(f) )
else:
    raise ValueError("args.raw_train_json_path should be str or list")

input_ids_list = []
labels_list = []

def process_single_data(item):
    """
    处理单个数据项，返回input_ids和labels
    """
    tok = TokenizerForMultiturns(item['messages'], tokenizer)
    input_ids, labels = tok.get_tokenization_res()
    return input_ids, labels

# 使用多进程并行处理数据
# 设置进程数，通常为CPU核心数
num_processes = min(32, os.cpu_count())  # 限制最大进程数

# 使用多进程池并行处理数据
with Pool(processes=num_processes) as pool:
    # 使用tqdm显示进度条
    results = list(tqdm(
        pool.imap(process_single_data, data), 
        total=len(data),
        desc="Processing data"
    ))

# 拆分结果
input_ids_list = [result[0] for result in results]
labels_list = [result[1] for result in results]
print('train data Num: ', len(input_ids_list))
# 计算最大长度
max_ = max(len(input_ids) for input_ids in input_ids_list)
print('raw data MAXI LENGTH: ' , max_)

dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list
    })

os.makedirs(args.train_data_save_folder, exist_ok=True)
dataset.save_to_disk(args.train_data_save_folder)



