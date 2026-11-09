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

import torch
from datasets import load_dataset
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from tqdm import tqdm

ds_collections = {
    'MathVista_testmini': {
        'root': 'AI4Math/MathVista',
        'max_new_tokens': 4096,
        'min_new_tokens': 1,
        'split': 'testmini'
    },
    'MathVista_test': {
        'root': 'AI4Math/MathVista',
        'max_new_tokens': 4096,
        'min_new_tokens': 1,
        'split': 'test'
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


def collate_fn(batches):
    images = [_['images'] for _ in batches]
    data_items = [_['data_item'] for _ in batches]
    return images, data_items


class MathVistaDataset(torch.utils.data.Dataset):

    def __init__(self, root, split):
        dataset = load_dataset(root, cache_dir=os.path.join(os.getcwd(), 'eval/vlm/data/MathVista/'))
        self.data = dataset[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        image = data_item['decoded_image']
        del data_item['decoded_image']

        images = [image.convert('RGB') if image.mode != 'RGB' else image]

        return {
            'images': images,
            'data_item': data_item,
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

    for ds_name in args.datasets:
        dataset = MathVistaDataset(
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

        outputs = []
        for _, (images, data_items) in tqdm(enumerate(dataloader)):
            if args.cot:
                question = COT_INSTRUCTION.format(question=data_items[0]['query'])
            else:
                question = data_items[0]['query']

            images = images[0]
            images, conversation = process_conversation(images, question)

            pred = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images,
                prompt=conversation,
                max_length=ds_collections[ds_name]['max_new_tokens'] if not args.cot else 4096, # TODO: how to use ds_collections[ds_name]['min_new_tokens']
            )

            data_item = data_items[0]
            data_item['response'] = pred
            outputs.append(data_item)

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            temp = {}
            for data_item in merged_outputs:
                pid = data_item['pid']
                temp[pid] = data_item

            print(f'Evaluating {ds_name} ...')
            results_file = 'results.json'
            output_path = os.path.join(args.out_dir, 'results.json')
            json.dump(temp, open(output_path, 'w'), indent=4)
            print('Results saved to {}'.format(output_path))

            if args.cot:
                cmd = f'python eval/vlm/eval/mathvista/extract_answer_mp.py --output_file {results_file} --output_dir {args.out_dir}'
            else:
                cmd = f'python eval/vlm/eval/mathvista/extract_answer_mp.py --output_file {results_file} --output_dir {args.out_dir}'
            print(cmd)
            os.system(cmd)

            cmd = f'python eval/vlm/eval/mathvista/calculate_score.py --output_file {results_file} --output_dir {args.out_dir} --score_file score.json'
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='MathVista_testmini')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cot', action='store_true')
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
