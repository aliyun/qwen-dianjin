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
import itertools
import json
import os
import random
import re

import torch
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'pope': {
        'root': 'eval/vlm/data/pope/val2014',
        'question': 'eval/vlm/data/pope/llava_pope_test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
        'min_new_tokens': 1,
    },
    'pope_cot': {
        'root': 'eval/vlm/data/pope/val2014',
        'question': 'eval/vlm/data/pope/llava_pope_test.jsonl',
        'metric': None,
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    }
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


def extract_answer(text):
    match = re.search(r'(Final answer:|Answer:)\s*(.*)', text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return text


def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversations = [_['conversations'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    return questions, images, conversations, question_ids, annotations


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root, data, prompt):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = json.loads(self.data[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'text'], data['question_id'], data.get('answer', None)

        image = os.path.join(self.root, image)
        image = Image.open(image).convert('RGB')
        images = [image]

        llava_prompt = 'Answer the question using a single word or phrase.'
        assert llava_prompt in question
        question = question.replace(llava_prompt, self.prompt).strip()

        if args.cot:
            question = COT_INSTRUCTION.format(question=question)
        
        images, conversation = process_conversation(images, question)

        return {
            'question_id': question_id,
            'question': question,
            'images': images,
            'conversations': conversation,
            'annotation': annotation
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
    prompt = '' if args.cot else 'Answer the question using a single word or phrase.'
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = VQADataset(
            root=ds_collections[ds_name]['root'],
            data=ds_collections[ds_name]['question'],
            prompt=prompt,
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

        outputs = []
        for _, (questions, images, conversations, question_ids, annotations) in tqdm(enumerate(dataloader)):
            pred = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images[0], # batch=1
                prompt=conversations[0], # batch=1
                max_length=ds_collections[ds_name]['max_new_tokens'], # TODO: how to use ds_collections[ds_name]['min_new_tokens']
            )
            pred_orig = pred
            if args.cot:
                pred = extract_answer(pred).strip()
            answers = [pred]

            for question_id, answer, annotation in zip(question_ids, answers, annotations):
                outputs.append({
                    'question_id': question_id,
                    'text': pred,
                    'text_orig': pred_orig,
                    'metadata': {},
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            results_file = os.path.join(args.out_dir, 'results.json')
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))
            cmd = 'python eval/vlm/eval/pope/eval_pope.py ' \
                  '--annotation-dir eval/vlm/data/pope/coco ' \
                  '--question-file eval/vlm/data/pope/llava_pope_test.jsonl ' \
                  f'--result-file {results_file} ' \
                  f'--out-dir {args.out_dir}'
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='pope')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--cot', action='store_true')
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
