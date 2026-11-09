from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="DeepSpeed step checkpoint dir (has pytorch_model/, latest)")
    p.add_argument("--out", required=True, help="output .pt path")
    p.add_argument("--prefix", default="language_model.", help="prefix to prepend to reconstructed keys")
    p.add_argument("--expect", type=int, default=926, help="expected trainable tensor count (sanity; 0=skip)")
    args = p.parse_args()

    import torch
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    _orig_load = torch.load
    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)
    torch.load = _load

    print(f"[consolidate] reading ZeRO shards from {args.ckpt} (trainable only) ...", flush=True)
    sd = get_fp32_state_dict_from_zero_checkpoint(args.ckpt, exclude_frozen_parameters=True)
    print(f"[consolidate] reconstructed {len(sd)} tensors", flush=True)

    sample = list(sd.keys())[:5]
    print("[consolidate] sample raw keys:", *sample, sep="\n  ")

    out = {}
    for k, v in sd.items():
        key = k if k.startswith(args.prefix) else args.prefix + k
        out[key] = v.to(torch.bfloat16).contiguous()
    print("[consolidate] sample prefixed keys:", *list(out.keys())[:5], sep="\n  ")

    if args.expect and len(out) != args.expect:
        print(f"[consolidate] WARNING: {len(out)} tensors != expected {args.expect}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    n_params = sum(v.numel() for v in out.values())
    print(f"[consolidate] saved {len(out)} tensors ({n_params/1e9:.2f}B params, bf16) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
