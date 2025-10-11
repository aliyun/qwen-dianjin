import re
from typing import Dict, List
import Levenshtein
import re

def normalized_formula(text):
    # Normalize math formulas before matching
    filter_list = ['\\mathbf', '\\mathrm', '\\mathnormal', '\\mathit', '\\mathbb', '\\mathcal', '\\mathscr', '\\mathfrak', '\\mathsf', '\\mathtt', 
                   '\\textbf', '\\text', '\\boldmath', '\\boldsymbol', '\\operatorname', '\\bm',
                   '\\symbfit', '\\mathbfcal', '\\symbf', '\\scriptscriptstyle', '\\notag',
                   '\\setlength', '\\coloneqq', '\\space', '\\thickspace', '\\thinspace', '\\medspace', '\\nobreakspace', '\\negmedspace',
                   '\\quad', '\\qquad', '\\enspace', '\\substackw', ' ']
                #    '\\left', '\\right', '{', '}', ' ']
    
    # delimiter_filter
    pattern = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
    match = pattern.search(text)

    if match:
        text = match.group(1).strip()
    
    tag_pattern = re.compile(r"\\tag\{.*?\}")
    text = tag_pattern.sub('', text)
    hspace_pattern = re.compile(r"\\hspace\{.*?\}")
    text = hspace_pattern.sub('', text)
    begin_pattern = re.compile(r"\\begin\{.*?\}")
    text = begin_pattern.sub('', text)
    end_pattern = re.compile(r"\\end\{.*?\}")
    text = end_pattern.sub('', text)
    col_sep = re.compile(r"\\arraycolsep.*?\}")
    text = col_sep.sub('', text)
    text = text.strip('.')
    
    for filter_text in filter_list:
        text = text.replace(filter_text, '')
        
    text = text.split("```latex")[-1].split("```")[0]
    text = text.lstrip("$$").rstrip("$$").strip()
    text = text.lower()
    return text

def calc_EditDist(pred,gt):
    try:
        norm_pred = normalized_formula(pred)
        norm_gt = normalized_formula(gt)
        upper_len = max(len(norm_pred), len(norm_gt))
        if len(norm_pred) == 0:
            return 1.0
        edit_dist = Levenshtein.distance(norm_pred, norm_gt)
        return edit_dist / upper_len
    except Exception as e:
        print(str(e))
        return 1.0


def extract_answer_content(predict):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, predict,re.DOTALL)
    if len(matches) == 0:
        return ''
    else:
        return matches[-1]
    
def extract_recog_content(predict):
    pattern = r"<recognition>(.*?)</recognition>"
    matches = re.findall(pattern, predict,re.DOTALL)
    if len(matches) == 0:
        return ''
    else:
        return matches[-1]


def grade_answer(predict: str, ground_truth: str) -> float:
    try:
        ed_dist = calc_EditDist(predict,ground_truth)
        return 1.0 - ed_dist
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

def toolcall_score(response):
    target_str_1 = 'get_extra_ocr_result()'
    target_str_2 = '<function>'
    target_str_3 = '</function>'
    return float( target_str_1 in response and target_str_2 in response and target_str_3 in response  )
    
def accuracy_reward_answer(predict: str, ground_truth: str) -> float:
    answer = extract_answer_content(predict)
    return grade_answer(answer,ground_truth) 



def compute_score_withTool(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    tool_weight = 0.05
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        # format score
        format_score = format_reward(predict)
        tool_score = toolcall_score(predict) #
        # final score
        accuracy_score = accuracy_reward_answer(predict, ground_truth) #  accuracy_reward_answer_with_recog(predict, ground_truth) 
        scores.append(
            {
                "overall": (1 - format_weight - tool_weight) * accuracy_score + format_weight * format_score + tool_weight * tool_score,
                "format": format_score,
                "toolcall": tool_score,
                "accuracy": accuracy_score,
            }
        )
    return scores


# Without Tool
def format_reward_purerl(response):
    must_shown_words = [
        "<answer>",
        "</answer>",
    ]
    for word in must_shown_words:
        if word not in response:
            return 0.0
    # Extract sections from response
    return 1.0

def compute_score(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        # format score
        format_score = format_reward_purerl(predict)
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
