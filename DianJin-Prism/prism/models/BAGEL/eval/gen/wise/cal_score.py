# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import argparse
from collections import defaultdict


def calculate_wiscore(consistency, realism, aesthetic_quality):
    return 0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality


WISE_CATEGORY_WEIGHTS = {
    'Cultural': 0.4,
    'Time': 0.167,
    'Space': 0.133,
    'Biology': 0.1,
    'Physics': 0.1,
    'Chemistry': 0.1,
}


def cal_culture(file_path):
    all_scores = []
    total_objects = 0
    has_9_9 = False
    
    with open(file_path, 'r') as file:
        for line in file:
            total_objects += 1
            data = json.loads(line)
            if 9.9 in [data['consistency'], data['realism'], data['aesthetic_quality']]:
                has_9_9 = True
            wiscore = calculate_wiscore(data['consistency'], data['realism'], data['aesthetic_quality'])
            all_scores.append(wiscore)
    
    if has_9_9 or total_objects < 400:
        print(f"Skipping file {file_path}: Contains 9.9 or has less than 400 objects.")
        return None
    
    total_score = sum(all_scores)
    avg_score = total_score / (len(all_scores)*2) if len(all_scores) > 0 else 0
    
    score = {
        'total': total_score,
        'average': avg_score
    }

    print(f"  Cultural - Total: {score['total']:.2f}, Average: {score['average']:.2f}")

    return avg_score


def cal_space_time(file_path):
    categories = defaultdict(list)
    total_objects = 0
    has_9_9 = False
    
    with open(file_path, 'r') as file:
        for line in file:
            total_objects += 1
            data = json.loads(line)
            if 9.9 in [data['consistency'], data['realism'], data['aesthetic_quality']]:
                has_9_9 = True
            subcategory = data['Subcategory']
            wiscore = calculate_wiscore(data['consistency'], data['realism'], data['aesthetic_quality'])
            if subcategory in ['Longitudinal time', 'Horizontal time']:
                categories['Time'].append(wiscore)
            else:
                categories['Space'].append(wiscore)
    
    if has_9_9 or total_objects < 300:
        print(f"Skipping file {file_path}: Contains 9.9 or has less than 300 objects.")
        return None
    
    total_scores = {category: sum(scores) for category, scores in categories.items()}
    avg_scores = {category: sum(scores) / (len(scores) * 2 )if len(scores) > 0 else 0 for category, scores in categories.items()}
    
    scores = {
        'total': total_scores,
        'average': avg_scores
    }

    print(f"  Time - Total: {scores['total'].get('Time', 0):.2f}, Average: {scores['average'].get('Time', 0):.2f}")
    print(f"  Space - Total: {scores['total'].get('Space', 0):.2f}, Average: {scores['average'].get('Space', 0):.2f}")

    return avg_scores


def cal_science(file_path):
    categories = defaultdict(list)
    total_objects = 0
    has_9_9 = False
    
    with open(file_path, 'r') as file:
        for line in file:
            total_objects += 1
            data = json.loads(line)
            if 9.9 in [data['consistency'], data['realism'], data['aesthetic_quality']]:
                has_9_9 = True
            
            prompt_id = data.get('prompt_id', 0)
            if 701 <= prompt_id <= 800:
                category = 'Biology'
            elif 801 <= prompt_id <= 900:
                category = 'Physics'
            elif 901 <= prompt_id <= 1000:
                category = 'Chemistry'
            else:
                category = "?"
            
            wiscore = calculate_wiscore(data['consistency'], data['realism'], data['aesthetic_quality'])
            categories[category].append(wiscore)
    
    if has_9_9 or total_objects < 300: 
        print(f"Skipping file {file_path}: Contains 9.9 or has less than 300 objects.")
        return None
    
    total_scores = {category: sum(scores) for category, scores in categories.items()}
    avg_scores = {category: sum(scores) / (len(scores)*2) if len(scores) > 0 else 0 for category, scores in categories.items()}

    scores = {
        'total': total_scores,
        'average': avg_scores
    }

    for category in ['Biology', 'Physics', 'Chemistry']:
        print(f"  {category} - Total: {scores['total'].get(category, 0):.2f}, Average: {scores['average'].get(category, 0):.2f}")
    
    return avg_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Quality Assessment Tool')
    parser.add_argument('--output_dir', required=True,
                        help='Path to the output directory')
    args = parser.parse_args()

    avg_score = dict()

    score = cal_culture(
        os.path.join(args.output_dir, "cultural_common_sense_scores.jsonl")
    )
    avg_score['Cultural'] = score

    scores = cal_space_time(
        os.path.join(args.output_dir, "spatio-temporal_reasoning_scores.jsonl")
    )
    avg_score.update(scores)

    scores = cal_science(
        os.path.join(args.output_dir, "natural_science_scores.jsonl")
    )
    avg_score.update(scores)

    avg_all = sum(avg_score[key] * weight for key, weight in WISE_CATEGORY_WEIGHTS.items())

    avg_score['Overall'] = avg_all
    keys = ""
    values = ""
    for k, v in avg_score.items():
        keys += f"{k} "
        values += f"{v:.2f} "
    print(keys)
    print(values)

    writer = open(os.path.join(args.output_dir, "results.txt"), 'w')
    print(f"write results to file {os.path.join(args.output_dir, 'results.txt')}")
    writer.write(keys + "\n")
    writer.write(values + "\n")
    writer.close()
