
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

_PRISM_ROOT = Path(__file__).resolve().parents[2]
_SENSENOVA_ROOT = _PRISM_ROOT / "prism" / "models" / "SenseNova-U1"
_SENSENOVA_RUNTIME = None


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_manifest(root: Path) -> dict:
    manifest_path = root / "runtime_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid official SenseNova runtime manifest: {manifest_path}") from exc
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"official SenseNova runtime manifest has no files: {manifest_path}")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            actual[str(path.relative_to(root))] = _file_hash(path)
    actual.pop("runtime_manifest.json", None)
    if actual != files:
        raise RuntimeError("official SenseNova runtime does not match runtime_manifest.json")
    return manifest


def _resolve_sensenova_runtime() -> tuple[str, Path, str]:
    global _SENSENOVA_RUNTIME
    if _SENSENOVA_RUNTIME is not None:
        return _SENSENOVA_RUNTIME
    if os.environ.get("SENSENOVA_U1_SRC"):
        raise RuntimeError("SENSENOVA_U1_SRC is unsupported; use SENSENOVA_U1_RUNTIME=official or prism")
    mode = os.environ.get("SENSENOVA_U1_RUNTIME", "official").strip().lower()
    if mode not in {"official", "prism"}:
        raise RuntimeError("SENSENOVA_U1_RUNTIME must be official or prism")
    if any(name == "sensenova_u1" or name.startswith("sensenova_u1.") for name in sys.modules):
        raise RuntimeError("cannot select SenseNova runtime after sensenova_u1 was imported")
    if mode == "official":
        root = _SENSENOVA_ROOT / "official"
        manifest = _official_manifest(root)
        revision = str(manifest.get("source_commit", "unknown"))
        source = root / "src"
    else:
        source = _SENSENOVA_ROOT / "src"
        revision = "prism"
    package = source / "sensenova_u1" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"SenseNova runtime package is missing: {package}")
    source = source.resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.find_spec("sensenova_u1")
    if spec is None or spec.origin is None or not _under(Path(spec.origin), source):
        raise RuntimeError(f"SenseNova runtime import did not resolve beneath {source}")
    _SENSENOVA_RUNTIME = (mode, source, revision)
    return _SENSENOVA_RUNTIME


def load_sensenova_u1(model_path, device="cuda", dtype="bfloat16"):
    mode, source, revision = _resolve_sensenova_runtime()
    import sensenova_u1
    from sensenova_u1.utils import load_model_and_tokenizer

    if not _under(Path(sensenova_u1.__file__), source):
        raise RuntimeError(f"SenseNova runtime imported from unexpected path: {sensenova_u1.__file__}")
    dt = getattr(torch, dtype)
    sensenova_u1.set_attn_backend("auto")
    print(f"[models] SenseNova runtime={mode} revision={revision} source={source} "
          f"attention={sensenova_u1.effective_attn_backend()}", flush=True)
    model, tok = load_model_and_tokenizer(model_path, dtype=dt, device=device)
    ckpt = os.environ.get("PRISM_TRAINABLE_CKPT")
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys in {ckpt}: {unexpected[:5]}"
        print(f"[models] overlaid prism ckpt: {len(sd)} tensors <- {ckpt}", flush=True)
    return model, tok


def generate_sensenova_u1(model, tokenizer, images, prompt, max_new_tokens=1024,
                          device="cuda", no_think=True):
    from sensenova_u1.models.neo_unify.utils import load_image_native

    pvs, ghs = [], []
    for img in images:
        pv, gh = load_image_native(img)
        pvs.append(pv)
        ghs.append(gh)
    pixel_values = torch.cat(pvs, dim=0).to(device, dtype=model.dtype)
    grid_hw = torch.cat(ghs, dim=0).to(device)
    gen_cfg = {"max_new_tokens": max_new_tokens, "do_sample": False}

    n_img = len(images)
    deficit = n_img - prompt.count("<image>")
    question = ("<image>\n" * deficit + prompt) if deficit > 0 else prompt

    if not no_think:
        with torch.inference_mode():
            response, _ = model.chat(
                tokenizer, pixel_values, question, gen_cfg,
                history=None, return_history=True, grid_hw=grid_hw,
            )
        return response

    from sensenova_u1.models.neo_unify.conversation import get_conv_template
    IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CTX)
    model.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START)

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt() + "<think>\n\n</think>\n\n"

    for i in range(grid_hw.shape[0]):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * model.downsample_ratio ** 2)
        image_tokens = IMG_START + IMG_CTX * num_patch_token + IMG_END
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    gen_cfg["eos_token_id"] = eos_token_id
    with torch.inference_mode():
        out = model.generate(
            pixel_values=pixel_values, input_ids=input_ids,
            grid_hw=grid_hw, attention_mask=attention_mask, **gen_cfg,
        )
    response = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return response.split(template.sep.strip())[0].strip()



_PRISM_ROOT = os.environ.get("PRISM_ROOT", str(Path(__file__).resolve().parents[2]))
if os.path.isdir(_PRISM_ROOT) and _PRISM_ROOT not in sys.path:
    sys.path.insert(0, _PRISM_ROOT)


def load_bagel(model_path, device="cuda", dtype="bfloat16"):
    from prism.adapters.registry import build_adapter
    adapter = build_adapter("bagel", model_path=model_path, device=device, dtype=dtype)
    return adapter, None


def generate_bagel(model, tokenizer, images, prompt, max_new_tokens=1024, device="cuda"):
    return model.understand(images, prompt, think=False, max_new_tokens=max_new_tokens)


MODELS = {
    "sensenova-u1": {"load": load_sensenova_u1, "generate": generate_sensenova_u1},
    "bagel": {"load": load_bagel, "generate": generate_bagel},
}


def _remote_generate_factory(url):
    import json
    import urllib.request

    def remote_generate(model, tokenizer, images, prompt, max_new_tokens=1024,
                        device="cuda"):
        for im in images:
            if not isinstance(im, str):
                raise TypeError(
                    "remote server mode only supports image paths, got "
                    f"{type(im).__name__}")
        body = json.dumps({"images": list(images), "prompt": prompt,
                           "max_new_tokens": max_new_tokens}).encode()
        req = urllib.request.Request(
            url + "/generate", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            out = json.loads(r.read())
        if "error" in out:
            raise RuntimeError(f"server {url}: {out['error']}")
        return out["response"]

    return remote_generate


def create_model(name, model_path, **kwargs):
    server = os.environ.get("EVAL_SERVER_URL")
    if server:
        server = server.rstrip("/")
        print(f"[models] remote server mode: {server} (no local load)", flush=True)
        return None, None, _remote_generate_factory(server)
    entry = MODELS.get(name)
    if entry is None:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    model, tokenizer = entry["load"](model_path, **kwargs)
    return model, tokenizer, entry["generate"]



_T2I_NORM_MEAN = (0.5, 0.5, 0.5)
_T2I_NORM_STD = (0.5, 0.5, 0.5)


def _t2i_to_pil(batch):
    import numpy as np
    from PIL import Image
    mean = torch.tensor(_T2I_NORM_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_T2I_NORM_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    arr = (batch.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def load_t2i(model_name, model_path, device="cuda", dtype="bfloat16"):
    if model_name.startswith("bagel"):
        adapter, _ = load_bagel(model_path, device=device, dtype=dtype)
        return ("bagel", adapter, None)
    model, tok = load_sensenova_u1(model_path, device=device, dtype=dtype)
    return ("sensenova", model, tok)


def t2i_images(backend, prompt, width, height, *, cfg_scale=4.0, cfg_norm="global",
               num_steps=50, seed=42, think=False, n_samples=1, device="cuda"):
    kind = backend[0]
    if kind == "sensenova":
        model, tok = backend[1], backend[2]
        with torch.inference_mode():
            out = model.t2i_generate(
                tok, prompt, image_size=(width, height),
                cfg_scale=cfg_scale, cfg_norm=cfg_norm, num_steps=num_steps,
                batch_size=n_samples, seed=seed, think_mode=think,
            )
        tensor = out[0] if think else out
        return _t2i_to_pil(tensor)
    if kind == "bagel":
        adapter = backend[1]
        return adapter.regenerate_images_batch(
            prompt, n_samples, image_size=(width, height),
            cfg_scale=cfg_scale, num_steps=num_steps, seed=seed, think=think,
        )
    raise ValueError(f"unknown T2I backend: {kind}")
