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
import numpy as np


def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    ret_message = ""
    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))
    ret_message += 'TP\tFP\tTN\tFN\t\n'
    ret_message += '{}\t{}\t{}\t{}\n'.format(TP, FP, TN, FN)

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))

    ret_message += 'Accuracy: {}\n'.format(acc)
    ret_message += 'Precision: {}\n'.format(precision)
    ret_message += 'Recall: {}\n'.format(recall)
    ret_message += 'F1 score: {}\n'.format(f1)
    ret_message += 'Yes ratio: {}\n'.format(yes_ratio)
    ret_message += '%.3f, %.3f, %.3f, %.3f, %.3f\n' % (f1, acc, precision, recall, yes_ratio)
    return f1, ret_message


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-dir', type=str)
    parser.add_argument('--question-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--out-dir', type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = json.loads(open(args.result_file).read())
    avg_f1 = []
    ret_message = ""
    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        ret_message += 'Category: {}, # samples: {}\n'.format(category, len(cur_answers))
        f1, ret = eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        ret_message += ret
        print('====================================')
        ret_message += '====================================\n'
        avg_f1.append(f1)
    print(f"Avg F1 score: {np.array(avg_f1).mean()}")
    ret_message += f"Avg F1 score: {np.array(avg_f1).mean()}\n"

    writer = open(os.path.join(args.out_dir, "results.txt"), 'w')
    print(f"write results to file {os.path.join(args.out_dir, 'results.txt')}")
    writer.write(ret_message)
    writer.close()
