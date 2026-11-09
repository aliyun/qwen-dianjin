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
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utilities import *
from prompts.ext_ans import demo_prompt

openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == '' or extraction is None:
        return False
    return True

def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f'{query}\n\n{response}'
    full_prompt = f'{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: '
    return full_prompt

def _extract_answer(text):
    match = re.search(r'(Final answer:|Answer:)\s*(.*)', text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return text

def extract_answer(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']

    if response == '':
        return ''

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == 'integer':
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == 'float':
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print('Quickly extracting answer...')
        try:
            result = _extract_answer(response)
            return result
        except:
            pass

    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = get_chat_response(full_prompt, openai.api_key, patience=5, model=args.llm_engine)
        return extraction
    except Exception as e:
        print(e)

    return ''

def process_problem(pid, results, label, args):
    problem = results[pid]
    response = problem[label]
    extraction = extract_answer(response, problem, args.quick_extract)
    return pid, extraction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--output_file', type=str, default='mathvista_answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    # model
    parser.add_argument('--llm_engine', type=str, default='gpt-4o-2024-11-20', help='llm engine',
                        choices=['gpt-3.5-turbo', 'gpt-3.5', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613',
                                 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20'])
    parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    parser.add_argument('--output_label', type=str, default='', help='label for the output file')
    parser.add_argument('--max_workers', type=int, default=40, help='max workers for ThreadPoolExecutor')
    args = parser.parse_args()

    label = args.response_label
    result_file = os.path.join(args.output_dir, args.output_file)

    if args.output_label != '':
        output_file = result_file.replace('.json', f'_{args.output_label}.json')
    else:
        output_file = result_file

    print(f'Reading {result_file}...')
    results = read_json(result_file)

    full_pids = list(results.keys())
    if args.number > 0:
        full_pids = full_pids[:min(args.number, len(full_pids))]
    print('Number of total problems:', len(full_pids))

    if args.rerun:
        test_pids = full_pids
    else:
        test_pids = []
        for pid in full_pids:
            if 'extraction' not in results[pid] or not verify_extraction(results[pid]['extraction']):
                test_pids.append(pid)

    test_num = len(test_pids)
    print('Number of problems to run:', test_num)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_pid = {}
        for pid in test_pids:
            future = executor.submit(process_problem, pid, results, label, args)
            future_to_pid[future] = pid

        completed_count = 0
        for future in tqdm(as_completed(future_to_pid), total=test_num):
            pid = future_to_pid[future]
            try:
                pid_result, extraction = future.result()
                results[pid_result]['extraction'] = extraction
            except Exception as e:
                print(f'Error processing pid={pid}: {e}')

            completed_count += 1
            if (completed_count % args.save_every == 0) or (completed_count == test_num):
                print(f'Saving results to {output_file}... [{completed_count}/{test_num}]')
                save_json(results, output_file)
                print('Results saved.')

    print('All done!')
