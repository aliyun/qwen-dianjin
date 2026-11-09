from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--r", type=int, required=True, help="LoRA rank")
    p.add_argument("--alpha", type=float, required=True, help="LoRA alpha (scaling = alpha/r)")
    args = p.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    raw_base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scaling = args.alpha / args.r
    print(f"[lora] scaling = alpha/r = {args.alpha}/{args.r} = {scaling}", flush=True)

    sd = torch.load(args.lora, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    def target_key(k: str) -> str | None:
        m = re.match(r"language_model\.base_model\.model\.(.+)\.lora_([AB])\.default\.weight", k)
        if m:
            return f"language_model.{m.group(1)}.weight"
        m = re.match(r"base_model\.model\.model\.(.+)\.lora_([AB])\.default\.weight", k)
        if m:
            return f"language_model.model.{m.group(1)}.weight"
        return None

    deltas: dict[str, dict] = {}
    for k, v in sd.items():
        tk = target_key(k)
        if tk is None:
            continue
        part = "A" if ".lora_A." in k else "B" if ".lora_B." in k else ""
        if part:
            deltas.setdefault(tk, {})[part] = v
    n_pairs = sum(1 for d in deltas.values() if "A" in d and "B" in d)
    print(f"[lora] {len(deltas)} target modules, {n_pairs} complete A/B pairs", flush=True)
    assert n_pairs == len(deltas), "some LoRA modules miss an A or B tensor"

    if raw_base.is_file():
        base_dir = raw_base.parent
        safetensors_files = [raw_base]
        single_name = raw_base.name
    else:
        base_dir = raw_base
        shards = sorted(base_dir.glob("model-*-of-*.safetensors"))
        if shards:
            safetensors_files = shards
            single_name = ""
        else:
            ema = base_dir / "ema.safetensors"
            assert ema.exists(), f"no model shards or ema.safetensors under {base_dir}"
            safetensors_files = [ema]
            single_name = "ema.safetensors"

    for f in base_dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out / f.name)

    merged = 0
    for src in safetensors_files:
        tensors = load_file(str(src))
        n = 0
        for key in list(tensors.keys()):
            if key in deltas:
                A = deltas[key]["A"].float()
                B = deltas[key]["B"].float()
                w = tensors[key]
                delta = (B @ A) * scaling
                assert delta.shape == w.shape, f"shape {key}: {tuple(delta.shape)} vs {tuple(w.shape)}"
                tensors[key] = (w.float() + delta).to(w.dtype).contiguous()
                n += 1
        dst_name = single_name or src.name
        save_file(tensors, str(out / dst_name), metadata={"format": "pt"})
        merged += n
        print(f"[lora] {src.name}: merged {n} tensors", flush=True)

    assert merged == len(deltas), f"merged {merged} != {len(deltas)} LoRA modules (base key mismatch?)"
    print(f"[lora] copied config/tokenizer/index files", flush=True)
    print(f"[lora] DONE: merged {merged} LoRA modules -> {out}", flush=True)


if __name__ == "__main__":
    main()
