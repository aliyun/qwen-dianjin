import re
from typing import Dict, List

def extract_answer_content(predict):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, predict,re.DOTALL)
    if len(matches) == 0:
        return ''
    else:
        return matches[-1]

def grade_answer(predict: str, ground_truth: str) -> float:
    try:
        if predict.strip() == ground_truth.strip():
            return 1.0
        else:
            return 0.0
    except:
        return 0.0

def format_reward(response):
    must_shown_words = [
        "<recognition>",
        "</recognition>",
        "<rethink>",
        "</rethink>",
        "<answer>",
        "</answer>",
    ]
    for word in must_shown_words:
        if word not in response:
            return 0.0
    # Extract sections from response
    return 1.0


def accuracy_reward_answer(predict: str, ground_truth: str) -> float:
    answer = extract_answer_content(predict)
    return grade_answer(answer,ground_truth) 


def compute_score_withTool(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        # format score
        format_score = format_reward(predict)    
        # final score
        accuracy_score = accuracy_reward_answer(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores

def format_reward_answer(response):
    must_shown_words = [
        "<answer>",
        "</answer>",
    ]
    for word in must_shown_words:
        if word not in response:
            return 0.0
    return 1.0

def compute_score(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        # format score
        format_score = format_reward_answer(predict)    
        # final score
        accuracy_score = accuracy_reward_answer(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores

