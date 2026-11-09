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
import csv
import itertools
import json
import os
import random

import torch
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'MMVP': {
        'root': 'eval/vlm/data/MMVP',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
    },
}


def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversations = [_['conversations'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    options = [_['option'] for _ in batches]
    return questions, images, conversations, answers, data_ids, options


class MMVPDataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt):
        # run for each subject
        meta_path = os.path.join(root, 'Questions.csv')
        with open(meta_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            data = [row for row in csv_reader]

        self.data = data
        self.root = root
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_id = data['lndex'] if 'lndex' in data else data['Index']
        question = data['Question']
        image = os.path.join(self.root + '/MMVP Images', data_id + '.jpg')
        image = Image.open(image).convert('RGB')

        options = data['Options'].split('(b)')
        options[0] = options[0].replace('(a)', '').strip()
        options[1] = options[1].replace('(b)', '').strip()

        answer = data['Correct Answer'] if 'Correct Answer' in data else None
        answer = answer.replace('(a)', 'A').replace('(b)', 'B').replace('(c)', 'C').replace('(d)', 'D')
        choice_list = []
        new_options = {}
        multiple_choices = ['A', 'B', 'C', 'D']
        for i, c in enumerate(options):
            choice_list.append('{}. {}'.format(multiple_choices[i], c.strip()))
            new_options[multiple_choices[i]] = c.strip()
        choice_txt = '\n'.join(choice_list)

        images = [image]

        if len(choice_txt) > 0:
            question += '\n' + choice_txt
        question += '\n' + self.prompt

        images, conversation = process_conversation(images, question)

        return {
            'question': question,
            'images': images,
            'conversations': conversation,
            'answer': answer,
            'option': new_options,
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


def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k

    return pred


def evaluate_chat_model():
    prompt = "Answer with the option's letter from the given choices directly."
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MMVPDataset(
            root=ds_collections[ds_name]['root'],  # hf dataset path.
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
        for _, (questions, images, conversations, answers, data_ids, options) in tqdm(enumerate(dataloader)):
            pred = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images[0], # batch=1
                prompt=conversations[0], # batch=1
                max_length=ds_collections[ds_name]['max_new_tokens'], # TODO: how to use ds_collections[ds_name]['min_new_tokens']
            )

            if len(options[0]) == 0:
                preds = [pred]
            else:
                preds = [post_process(pred, options[0])]

            for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'data_id': data_id
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            results_file = 'results.jsonl'
            output_path = os.path.join(args.out_dir, results_file)
            writer = open(output_path, 'w')

            num_correct, num_total = 0, 0
            index, round_correct = 0, 0
            for item in merged_outputs:
                writer.write(json.dumps(item) + '\n')
                answer = item['answer']
                gt_answer = item['gt_answers']

                index += 1

                if answer == gt_answer:
                    round_correct += 1
                if index == 2:
                    index = 0
                    if round_correct == 2:
                        num_correct += 1
                    round_correct = 0

                    num_total += 1

            writer.close()
            print('Results saved to {}'.format(output_path))
            print(f'The accuracy is {num_correct/num_total}')

            results_file = 'results.txt'
            output_path = os.path.join(args.out_dir, results_file)
            writer = open(output_path, 'w')
            writer.write(f'The accuracy is {num_correct/num_total}')
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='MMVP')
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
