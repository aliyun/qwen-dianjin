import json
import os
import argparse
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(description="Evaluate tool-call metrics with selectable data source.")
parser.add_argument("--eval_data_path", type=str, required=True,help="file path of eval data")
parser.add_argument("--drop_error_data", type=bool, default=True,help="whether to drop error data")
args = parser.parse_args()
    
def extract_toolcall_from_meg(messages):
    # only tool names are required. 
    # TODO YOU CAN ALSO EVALUATE `TOOL ARGS`.
    toolcall_list = []
    for turn in messages:
        if 'tool_calls' in turn and turn['tool_calls']:
            for tool_call in turn['tool_calls']:
                toolcall_list.append(str(tool_call['function']['name']))
    return toolcall_list

def calc_tp_fp_fn(pred_list, gt_list):
    # Calculate Tool Recall. Tool Precision. Tool F1
    pred_counter = Counter(pred_list)
    gt_counter = Counter(gt_list)
    TP = sum((pred_counter & gt_counter).values())
    FP = sum((pred_counter - gt_counter).values())
    FN = sum((gt_counter - pred_counter).values())
    return TP, FP, FN

def get_instance_result(data_dict, model_name_key='messages_pred'):
    """
    Necessary Keys in data_dict:
      'scenery', 'diff_level.level', 'messages'(ground truth),  'model_name_key'(model resp to be evaluated)
    """
    class_Level1 = data_dict['scenery'].split('-')[0].strip()
    class_difficulty = data_dict['diff_level']['level']
    task = data_dict.get('task','any').replace(' ', '_') # single_step / multi_step / multi_turn

    gt_messages = data_dict['messages']
    gt_toolcalls = extract_toolcall_from_meg(gt_messages)

    if model_name_key in data_dict:
        model_messages = data_dict[model_name_key]
    else:
        raise ValueError(f"{model_name_key} not found in data_dict")
    model_toolcalls = extract_toolcall_from_meg(model_messages)

    TP, FP, FN = calc_tp_fp_fn(pred_list=model_toolcalls, gt_list=gt_toolcalls)
    consist = int(gt_toolcalls == model_toolcalls)  
    return {'class_Level1':class_Level1,
            'class_difficulty':class_difficulty,
            'task':task,
            'TP':TP,
            'FP':FP,
            'FN':FN,
            'consist':consist}

def safe_div(a, b):
    return a / b if b > 0 else 0

def compute_metrics(records):
    precisions, recalls, f1s = [], [], []
    consist_list = []

    total_TP, total_FP, total_FN = 0, 0, 0

    group_level1 = defaultdict(list)
    group_difficulty = defaultdict(list)
    group_task = defaultdict(list)

    for item_dict in records:
        class_lvl1, class_diff, task = item_dict['class_Level1'], item_dict['class_difficulty'], item_dict['task']
        TP, FP, FN, consist = item_dict['TP'], item_dict['FP'], item_dict['FN'], item_dict['consist']
        p = safe_div(TP, TP + FP)
        r = safe_div(TP, TP + FN)
        f1 = safe_div(2 * p * r, p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        consist_list.append(consist)

        total_TP += TP
        total_FP += FP
        total_FN += FN

        group_level1[class_lvl1].append((TP, FP, FN, consist))
        group_difficulty[class_diff].append((TP, FP, FN, consist))
        group_task[task].append((TP, FP, FN, consist))

    macro_P = round(sum(precisions) / len(precisions), 4) if precisions else 0
    macro_R = round(sum(recalls) / len(recalls), 4) if recalls else 0
    macro_F1 = round(sum(f1s) / len(f1s), 4) if f1s else 0

    micro_P = safe_div(total_TP, total_TP + total_FP)
    micro_R = safe_div(total_TP, total_TP + total_FN)
    micro_F1 = safe_div(2 * micro_P * micro_R, micro_P + micro_R)

    overall = {
        "macro_Precision": macro_P,
        "macro_Recall": macro_R,
        "macro_F1": macro_F1,
        "micro_Precision": round(micro_P, 4),
        "micro_Recall": round(micro_R, 4),
        "micro_F1": round(micro_F1, 4),
        "N": len(records),
        "consist_mean": round(sum(consist_list) / len(consist_list), 4) if consist_list else 0
    }

    level1_metrics = {}
    for k, values in group_level1.items():
        TP_sum = sum(v[0] for v in values)
        FP_sum = sum(v[1] for v in values)
        FN_sum = sum(v[2] for v in values)
        consist_vals = sum(v[3] for v in values)

        micro_P = safe_div(TP_sum, TP_sum + FP_sum)
        micro_R = safe_div(TP_sum, TP_sum + FN_sum)
        micro_F1 = safe_div(2 * micro_P * micro_R, micro_P + micro_R)

        ps, rs, f1s = [], [], []
        for TP, FP, FN, _ in values:
            p = safe_div(TP, TP + FP)
            r = safe_div(TP, TP + FN)
            f1 = safe_div(2 * p * r, p + r)
            ps.append(p); rs.append(r); f1s.append(f1)

        level1_metrics[k] = {
            "macro_Precision": round(sum(ps)/len(ps), 4) if ps else 0,
            "macro_Recall": round(sum(rs)/len(rs), 4) if rs else 0,
            "macro_F1": round(sum(f1s)/len(f1s), 4) if f1s else 0,
            "micro_Precision": round(micro_P, 4),
            "micro_Recall": round(micro_R, 4),
            "micro_F1": round(micro_F1, 4),
            "N": len(values),
            "consist_mean": round(consist_vals/len(values), 4) if values else 0
        }

    difficulty_metrics = {}
    for k, values in group_difficulty.items():
        TP_sum = sum(v[0] for v in values)
        FP_sum = sum(v[1] for v in values)
        FN_sum = sum(v[2] for v in values)
        consist_vals = sum(v[3] for v in values)

        micro_P = safe_div(TP_sum, TP_sum + FP_sum)
        micro_R = safe_div(TP_sum, TP_sum + FN_sum)
        micro_F1 = safe_div(2 * micro_P * micro_R, micro_P + micro_R)

        ps, rs, f1s = [], [], []
        for TP, FP, FN, _ in values:
            p = safe_div(TP, TP + FP)
            r = safe_div(TP, TP + FN)
            f1 = safe_div(2 * p * r, p + r)
            ps.append(p); rs.append(r); f1s.append(f1)

        difficulty_metrics[k] = {
            "macro_Precision": round(sum(ps)/len(ps), 4) if ps else 0,
            "macro_Recall": round(sum(rs)/len(rs), 4) if rs else 0,
            "macro_F1": round(sum(f1s)/len(f1s), 4) if f1s else 0,
            "micro_Precision": round(micro_P, 4),
            "micro_Recall": round(micro_R, 4),
            "micro_F1": round(micro_F1, 4),
            "N": len(values),
            "consist_mean": round(consist_vals/len(values), 4) if values else 0
        }

    task_metrics = {}
    for k, values in group_task.items():
        TP_sum = sum(v[0] for v in values)
        FP_sum = sum(v[1] for v in values)
        FN_sum = sum(v[2] for v in values)
        consist_vals = sum(v[3] for v in values)

        micro_P = safe_div(TP_sum, TP_sum + FP_sum)
        micro_R = safe_div(TP_sum, TP_sum + FN_sum)
        micro_F1 = safe_div(2 * micro_P * micro_R, micro_P + micro_R)

        ps, rs, f1s = [], [], []
        for TP, FP, FN, _ in values:
            p = safe_div(TP, TP + FP)
            r = safe_div(TP, TP + FN)
            f1 = safe_div(2 * p * r, p + r)
            ps.append(p); rs.append(r); f1s.append(f1)

        task_metrics[k] = {
            "macro_Precision": round(sum(ps)/len(ps), 4) if ps else 0,
            "macro_Recall": round(sum(rs)/len(rs), 4) if rs else 0,
            "macro_F1": round(sum(f1s)/len(f1s), 4) if f1s else 0,
            "micro_Precision": round(micro_P, 4),
            "micro_Recall": round(micro_R, 4),
            "micro_F1": round(micro_F1, 4),
            "N": len(values),
            "consist_mean": round(consist_vals/len(values), 4) if values else 0
        }
        
    return overall, level1_metrics, difficulty_metrics, task_metrics

def print_dict(d):
    for k, v in d.items():
        print(k, v)

def main():
    with open(args.eval_data_path,'r') as f:
        data = json.load(f)
    print('eval data Num: {} ... START ...'.format(len(data)))
    if args.drop_error_data:
        data = [data_dict for data_dict in data if not data_dict.get('error',False)]
        print('After drop error samples: {}'.format(len(data)))
    
    record = []
    for data_dict in data:
        try:
            data_processed_list = get_instance_result(data_dict)
        except KeyError as e:
            print(str(e))
            continue
        record.append(data_processed_list)

    print('{} 总处理数：{}'.format(  args.eval_data_path, len(record) ) )
    overall, level1_result, diff_result, task_result = compute_metrics(record)
    print('=' * 20)
    print_dict(overall)
    print('-' * 10 + '\n')
    print_dict(level1_result)
    print('-' * 10 + '\n')
    print_dict(diff_result)
    print('-' * 10 + '\n')
    print_dict(task_result)
    print('\n')
        
        

if __name__ == "__main__":
    main()