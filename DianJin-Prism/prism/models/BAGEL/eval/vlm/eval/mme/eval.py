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
import os
import re

from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='eval/vlm/eval/mme/Your_Results')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    args = parser.parse_args()

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    os.makedirs(args.out_dir, exist_ok=True)
    prompt = 'Answer the question using a single word or phrase.'

    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(args.out_dir, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            question = question + ' ' + prompt
            img_path = os.path.join('eval/vlm/data/mme/MME_Benchmark_release_version', filename, img)
            if not os.path.exists(img_path):
                img_path = os.path.join('eval/vlm/data/mme/MME_Benchmark_release_version', filename, "images", img)
            if not os.path.exists(img_path):
                continue
            images = [Image.open(img_path).convert('RGB')]
            images, conversation = process_conversation(images, question)

            response = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images,
                prompt=conversation,
                max_length=20,
            )
            response = post_processing(response)
            print(img, question, gt, response, sep='\t', file=fout)
        fin.close()
        fout.close()

    os.system(f"python -m eval.vlm.eval.mme.calculation --out-dir {args.out_dir}")
