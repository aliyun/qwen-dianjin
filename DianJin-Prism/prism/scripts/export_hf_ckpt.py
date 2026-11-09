from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--trained", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading trained sd: {args.trained}", flush=True)
    trained = torch.load(args.trained, map_location="cpu", weights_only=True)
    print(f"[export] {len(trained)} trained tensors", flush=True)

    replaced = 0
    shards = sorted(base.glob("model-*-of-*.safetensors"))
    assert shards, f"no safetensors shards under {base}"
    for shard in shards:
        sd = load_file(str(shard))
        n = 0
        for k in list(sd.keys()):
            if k in trained:
                t = trained[k].to(sd[k].dtype)
                assert t.shape == sd[k].shape, f"shape mismatch {k}: {t.shape} vs {sd[k].shape}"
                sd[k] = t.contiguous()
                n += 1
        save_file(sd, str(out / shard.name), metadata={"format": "pt"})
        replaced += n
        print(f"[export] {shard.name}: replaced {n}/{len(sd)} tensors", flush=True)

    missing = len(trained) - replaced
    if missing:
        in_base = set()
        for shard in shards:
            in_base.update(load_file(str(shard)).keys())
        leftover = [k for k in trained if k not in in_base]
        raise RuntimeError(f"{missing} trained tensors not found in base shards, e.g. {leftover[:5]}")

    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    print(f"[export] copied config/tokenizer/index files", flush=True)
    print(f"[export] DONE: {replaced} tensors replaced -> {out}", flush=True)


if __name__ == "__main__":
    main()
