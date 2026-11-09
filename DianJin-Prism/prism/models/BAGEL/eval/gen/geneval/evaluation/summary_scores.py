# Copyright (c) 2023 Dhruba Ghosh
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/djghosh13/geneval/blob/main/LICENSE.
#
# This modified file is released under the same license.

import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results

df = pd.read_json(args.filename, orient="records", lines=True)

# Measure overall success

print("Summary")
print("=======")
print(f"Total images: {len(df)}")
print(f"Total prompts: {len(df.groupby('metadata'))}")
print(f"% correct images: {df['correct'].mean():.2%}")
print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
print()

# By group

task_scores = []

print("Task breakdown")
print("==============")
for tag, task_df in df.groupby('tag', sort=False):
    task_scores.append(task_df['correct'].mean())
    print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
print()

print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")


print("\n\n==============")
output_info = "SO   TO   CT   CL   POS  ATTR ALL\n"
for score in task_scores:
    output_info += f"{score:.2f} "
output_info += f"{np.mean(task_scores):.2f}" + "\n"
print(output_info)
with open(os.path.join(os.path.dirname(args.filename), "geneval_results.txt"), "w") as f:
    f.write(output_info)