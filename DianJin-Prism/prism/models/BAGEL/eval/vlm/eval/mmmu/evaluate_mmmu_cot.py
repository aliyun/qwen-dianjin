# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import argparse
import json
import os
import random

import torch
from .data_utils import CAT_SHORT2LONG, process_single_sample
from datasets import concatenate_datasets, load_dataset
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'MMMU_validation_cot': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 16384,
        'min_new_tokens': 1,
        'split': 'validation'
    },
}

COT_INSTRUCTION = (
    'Your task is to answer the question below. '
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    '\n\n'
    'Question:'
    '\n\n'
    '{question}'
)

COT_OPEN_INSTRUCTION = (
    'Question: {question}'
)

COT_MC_INSTRUCTION = (
    'Question: {question} Options: {options}\n'
    'Try to reason about the question step by step to help you get the correct answer. '
    'You might find that sometimes no reasoning is needed if the answer is straightforward. '
    'Sometimes listing out a few reasoning steps will be helpful. '
    'In any case, please keep the reasoning concise.\n\n'

    'First, please describe what is included in the image. '
    'Then, respond with your reason first and output the final answer in the format "Final Answer: <answer>" where <answer> is the single correct letter choice A, B, C, D, E, F, etc, when options are provided. '
    'If you find that your reasoning lead to none of the choice, reject your reasoning and choose the most likely answer. '
    'You have to answer with one of the choices. If no options are provided, <answer> is your answer. '
    'If you would like to skip reasoning, just directly output the "Final Answer" part.'
)

COT_OPEN_INSTRUCTION_V2 = (
    "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here." +\
    "{question}" + "\nAnswer the question using a single word or phrase."
)

COT_MC_INSTRUCTION_V2 = (
    "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here." +\
    "Question: {question} Options: {options} " + "\nAnswer with the option's letter from the given choices directly."
)


def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversation = [_['conversation'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    options = [_['option'] for _ in batches]
    return questions, images, conversation, answers, data_ids, options


class MMMUDataset(torch.utils.data.Dataset):

    def __init__(self, root, split):
        # run for each subject
        sub_dataset_list = []
        for subject in tqdm(CAT_SHORT2LONG.values()):
            sub_dataset = load_dataset(root, subject, split=split, cache_dir=os.path.join(os.getcwd(), 'eval/vlm/data/MMMU/'))
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.data = concatenate_datasets(sub_dataset_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = process_single_sample(self.data[idx])
        data_id = data['id']
        question = data['question'].strip()
        pil_images = data['image']
        question_type = data['question_type'] # "open", "multiple-choice"

        choices = eval(data['options'])
        answer = data['answer'] if 'answer' in data else None

        choice_list = []
        options = {}
        multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c.strip()))
            options[multiple_choices[i]] = c.strip()
        choice_txt = '\n'.join(choice_list)
        images = []
        for idx, pil_image in enumerate(pil_images):
            if pil_image is not None:
                if idx == 0:
                    pil_image = pil_image.resize((pil_image.width * 2, pil_image.height * 2), Image.BILINEAR)
                images.append(pil_image)

        if len(choice_txt) > 0:
            question = COT_MC_INSTRUCTION_V2.format(question=question.strip(), options=choice_txt.strip())
        else:
            question = COT_OPEN_INSTRUCTION_V2.format(question=question.strip())

        # NOTE: Do not add <image> since <image 1> has been added
        # question = "<image>" * len(images) + "\n" + question

        images, conversation = process_conversation(images, question)

        return {
            'question': question,
            'images': images,
            'conversation': conversation,
            'answer': answer,
            'option': options,
            'data_id': data_id
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    random.seed(args.seed)
    output_path = os.path.join(args.out_dir, "results.jsonl")

    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            lines = f.readlines()
            for line in lines:
                processed_ids.add(json.loads(line)["data_id"])

    writer = open(output_path, 'a')

    for ds_name in args.datasets:
        dataset = MMMUDataset(
            root=ds_collections[ds_name]['root'],
            split=ds_collections[ds_name]['split'],
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        for _, (questions, images, conversation, answers, data_ids, options) in tqdm(enumerate(dataloader)):
            if data_ids[0] in processed_ids:
                continue

            pred = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images[0], # batch=1
                prompt=conversation[0], # batch=1
                max_length=ds_collections[ds_name]['max_new_tokens'], # TODO: how to use ds_collections[ds_name]['min_new_tokens']
            )
            preds = [pred.strip()]

            for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
                cur_output = {
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'data_id': data_id
                }
                writer.write(json.dumps(cur_output) + '\n')

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='MMMU_validation')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    evaluate_chat_model()
