import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def merge_lora(
    base_model_path: str,
    lora_checkpoint_path: str,
    output_dir: str,
    device: str = "cpu"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading base model from {base_model_path} ...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map=device if device != "auto" else None)
    lora_model = PeftModel.from_pretrained(base_model,lora_checkpoint_path , device_map=device if device != "auto" else None, ignore_generation=True)
    base_model = lora_model.merge_and_unload()
    print(f"Saving merged model to {output_dir} ...")
    base_model.save_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoints into a base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model directory or pretrained model name"
    )
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model"
    )
    # 可选参数
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to load and merge models on (default: cpu)"
    )
    args = parser.parse_args()
    merge_lora(
        base_model_path=args.base_model_path,
        lora_checkpoint_path=args.lora_checkpoint_path,
        output_dir=args.output_dir,
        device=args.device
    )
