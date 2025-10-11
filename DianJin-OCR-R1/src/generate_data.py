import os
import json
import base64
import argparse
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

from prompts import *


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/examples.json")
    parser.add_argument("--image_dir", type=str, default="data/images/train")
    parser.add_argument("--save_path", type=str, default="data/distill.json")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--task_type", type=str, default="seal")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    return args


def encode_image(image):
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def inference(args, client, prompt1, prompt2, image_name):
    try:
        base64_image = encode_image(os.path.join(args.image_dir, image_name))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {
                        "type": "text",
                        "text": prompt1,
                    },
                ],
            }
        ]

        res = client.chat.completions.create(model=args.model_name, messages=messages)
        content1 = res.choices[0].message.content

        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content1,
                    }
                ],
            }
        )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt2,
                    }
                ],
            }
        )
        res = client.chat.completions.create(model=args.model_name, messages=messages)
        content2 = res.choices[0].message.content
        return content1, content2
    except Exception as e:
        print(e)
        return None, None


def main():
    args = init_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.task_type == "seal":
        prompt1 = seal_prompt_1
        prompt2 = seal_prompt_2
    elif args.task_type == "table":
        prompt1 = table_prompt_1
        prompt2 = table_prompt_2
    elif args.task_type == "formula":
        prompt1 = formula_prompt_1
        prompt2 = formula_prompt_2
    else:
        raise NotImplemented("Invalid task type!")

    data = json.load(open(args.data_path))

    inputs = []
    for item in data:
        inputs.append(
            {
                "image_name": item["image_name"],
                "gt": item["gt"],
                "prompt1": prompt1,
                "prompt2": prompt2.format(tool1=item["tool1"], tool2=item["tool2"]),
            }
        )

    outputs = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        for i in tqdm(range(0, len(inputs), args.num_workers)):
            futures = {
                executor.submit(
                    inference,
                    args,
                    client,
                    inputs[i]["prompt1"],
                    inputs[i]["prompt2"],
                    inputs[i]["image_name"],
                ): item
                for item in inputs[i : i + args.num_workers]
            }
            for future in concurrent.futures.as_completed(futures):
                content1, content2 = future.result()
                item = futures[future]
                sample = {
                    "image_name": item["image_name"],
                    "gt": item["gt"],
                    "content1": content1,
                    "content2": content2,
                }
                outputs.append(sample)
    json.dump(outputs, open(args.save_path, "w"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
