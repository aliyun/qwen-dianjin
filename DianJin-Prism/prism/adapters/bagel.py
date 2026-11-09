
from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from ..core.protocol import _clear_markers
from ..core.types import InterleaveStep, Step, Trajectory
from .base import UMMAdapter
from .registry import register


def _ensure_on_path(repo_paths: Optional[dict[str, str]] = None) -> None:
    src = (
        (repo_paths or {}).get("bagel")
        or os.environ.get("BAGEL_SRC")
        or str(Path(__file__).resolve().parents[1] / "models" / "BAGEL")
    )
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)

_USER_IMAGE_LINE_RE = re.compile(r"^[ \t]*Image-\d+:[ \t]*<image\d*>[ \t]*\n?", re.MULTILINE)





_NUMBERED_HEADING_RE = re.compile(
    r"^\s*\*{1,2}\s*(scene|step|panel|shot|frame)\s*\d*\s*[:：]?[^\n]*$", re.IGNORECASE
)
_OPENING_RE = re.compile(
    r"^\s*(here is|here's|this is)\s+(the\s+)?(illustrated\s+)?answer[^\n]*\n?", re.IGNORECASE
)


_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_PROCESS_RE = re.compile(
    r"\b(cook|stir|mix|boil|simmer|fry|saute|heat|pour|add|cover|let|wait|bake|roast|"
    r"season|serve|repeat|do this|until (done|ready))\b[^.!?]{0,60}[.!?]?",
    re.IGNORECASE,
)


def _strip_rollout_scaffold(passage: str) -> str:
    s = "\n".join(l for l in (passage or "").split("\n")
                  if not _NUMBERED_HEADING_RE.match(l))
    s = _OPENING_RE.sub("", s)
    s = _BULLET_RE.sub("; ", s)
    s = _PROCESS_RE.sub("", s)
    s = re.sub(r"[;,\s]+([.;:!?])", r"\1", s)
    return s.strip(" ;,.").strip()


def _normalize_markers(body: str) -> str:
    return re.sub(r"\**<image>\**", "<image>", body or "")


def _clip_marker_text(body: str, max_images: int) -> tuple[str, int]:
    segments = (body or "").split("<image>")
    n = min(len(segments) - 1, max_images)
    if n <= 0:
        return body or "", 0
    if n < len(segments) - 1:
        return "<image>".join(segments[:n]) + "<image>", n
    return "<image>".join(segments[: n + 1]), n


def _shingle_overlap(a: str, b: str, n: int = 6) -> float:
    words_a = re.findall(r"[a-z]+", (a or "").lower())
    words_b = re.findall(r"[a-z]+", (b or "").lower())
    sa = {" ".join(words_a[i : i + n]) for i in range(max(0, len(words_a) - n + 1))}
    sb = {" ".join(words_b[i : i + n]) for i in range(max(0, len(words_b) - n + 1))}
    if not sa or not sb:
        return 1.0 if (a or "").strip() == (b or "").strip() else 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _bagel_text(s: str) -> str:
    s = _USER_IMAGE_LINE_RE.sub("", s or "")
    return _clear_markers(s)



_STUB = (
    "BagelAdapter is a model-agnostic proof stub: the {what} path is not wired to "
    "BAGEL weights yet. The Prism core accepts this adapter unchanged; implement "
    "this method against the BAGEL backend (SigLIP und expert + FLUX-VAE gen expert "
    "+ rectified-flow head) to activate it. See the module docstring for the mapping."
)


class _PlanBuilder:

    _ITEM = {"enable_cfg": 0, "special_token_loss": 0, "special_token_label": None}

    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer
        self.plan: list[dict] = []
        self.text_ids: list[list[int]] = []
        self.images: list = []
        self.ce_tags: list[tuple[str, int]] = []
        self.mse_tags: list[tuple[str, int]] = []

    def text(self, s: str, *, loss: int = 0, tag: str = "keep",
             image_marker: bool = False) -> None:
        if not s and not loss:
            return
        ids = self._tok.encode(s or "")
        if image_marker and loss:
            ids = ids + self._tok.encode("<image>")
        self.text_ids.append(ids)
        self.plan.append({"type": "text", "loss": int(loss), **self._ITEM})
        if loss:
            self.ce_tags.append((tag, len(ids) + 1))

    def vit(self, tensor) -> None:
        self.images.append(tensor)
        self.plan.append({"type": "vit_image", "loss": 0, **self._ITEM})

    def vae(self, tensor, *, loss: int, tag: str = "real") -> None:
        self.images.append(tensor)
        self.plan.append({"type": "vae_image", "loss": int(loss), **self._ITEM})
        if loss:
            h, w = tensor.shape[1] // 16, tensor.shape[2] // 16
            self.mse_tags.append((tag, h * w))

    @property
    def has_vit(self) -> bool:
        return any(p["type"] == "vit_image" for p in self.plan)

    @property
    def has_gen(self) -> bool:
        return any(p["type"] == "vae_image" and p["loss"] == 1 for p in self.plan)

    def sample(self) -> dict:
        return dict(
            image_tensor_list=list(self.images),
            text_ids_list=[list(t) for t in self.text_ids],
            sequence_plan=[dict(p) for p in self.plan],
            num_tokens=0,
        )


@register("bagel")
def _build(**kwargs: Any) -> "BagelAdapter":
    return BagelAdapter(**kwargs)


class BagelAdapter(UMMAdapter):

    def __init__(
        self,
        model_path: str = "",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        repo_paths: Optional[dict[str, str]] = None,
        gen_defaults: Optional[dict] = None,
        trainable: bool = False,
        train_defaults: Optional[dict] = None,
        **_: Any,
    ) -> None:
        self.model_path = model_path
        self._device = device
        self._dtype = dtype
        self._repo_paths = repo_paths or {}
        self._gen_defaults = {
            "cfg_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": (0.4, 1.0),
            "timestep_shift": 3.0,
            "num_steps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global",
            "image_size": (1024, 1024),
            "max_new_tokens": 1000,
            "text_temperature": 0.3,
            **(gen_defaults or {}),
        }
        self._train_cfg = {
            "ce_weight": 1.0,
            "mse_weight": 1.0,
            "cfg_txt_drop_prob": 0.1,
            "unlike_clamp": 0.9,
            "lambda_neg": 0.05,
            **(train_defaults or {}),
        }
        self.model = None
        self._tokenizer = None
        self._inferencer = None
        self._trainable = bool(trainable)
        self._train_ready = False
        self._vae = None
        self._vae_transform = None
        self._vit_transform = None
        self._new_token_ids = None
        self._pack_shim = None
        if self._trainable and model_path:
            self._ensure_trainable()

    def _ensure(self):
        if self._inferencer is not None:
            return self._inferencer
        if self._trainable:
            return self._ensure_trainable()
        _ensure_on_path(self._repo_paths)
        import torch
        from accelerate import (
            infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights,
        )
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer

        mp = self.model_path
        llm_config = Qwen2Config.from_json_file(os.path.join(mp, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(mp, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        vae_model, vae_config = load_ae(local_path=os.path.join(mp, "ae.safetensors"))
        config = BagelConfig(
            visual_gen=True, visual_und=True,
            llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
            vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
            latent_patch_size=2, max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        tokenizer = Qwen2Tokenizer.from_pretrained(mp)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        _max_mem = os.environ.get("BAGEL_MAX_MEM", "80GiB")
        device_map = infer_auto_device_map(
            model,
            max_memory={i: _max_mem for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        same_device = [
            "language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
            "vae2llm", "llm2vae", "connector", "vit_pos_embed",
        ]
        first = device_map.get(same_device[0], "cuda:0")
        for k in same_device:
            if k in device_map or torch.cuda.device_count() == 1:
                device_map[k] = first
        offload_dir = os.environ.get("BAGEL_OFFLOAD", "/tmp/bagel_offload")
        os.makedirs(offload_dir, exist_ok=True)
        model = load_checkpoint_and_dispatch(
            model, checkpoint=os.path.join(mp, "ema.safetensors"),
            device_map=device_map, offload_buffers=True, offload_folder=offload_dir,
            dtype=torch.bfloat16, force_hooks=True,
        ).eval()

        self.model = model
        self._patch_prepare_outputs_to_device()
        self._tokenizer = tokenizer
        self._new_token_ids = new_token_ids
        self._vae_transform = vae_transform
        self._vit_transform = vit_transform
        vae_model = vae_model.to(self._device).eval()
        vae_model.requires_grad_(False)
        self._vae = vae_model
        self._inferencer = InterleaveInferencer(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            vae_transform=vae_transform, vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        return self._inferencer

    def _ensure_trainable(self):
        if self._train_ready:
            return self._inferencer
        if self._inferencer is not None:
            raise RuntimeError(
                "BagelAdapter was already loaded in inference (accelerate-dispatch) "
                "mode; construct the adapter with trainable=True BEFORE any inference "
                "call to enable the training path."
            )
        _ensure_on_path(self._repo_paths)
        import torch
        from safetensors.torch import load_file
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer

        mp = self.model_path
        llm_config = Qwen2Config.from_json_file(os.path.join(mp, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(mp, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        vae_model, vae_config = load_ae(local_path=os.path.join(mp, "ae.safetensors"))
        config = BagelConfig(
            visual_gen=True, visual_und=True,
            llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
            vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
            latent_patch_size=2, max_latent_size=64,
        )
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        tokenizer = Qwen2Tokenizer.from_pretrained(mp)
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        if num_new_tokens > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)

        sd = load_file(os.path.join(mp, "ema.safetensors"), device="cpu")
        sd.pop("latent_pos_embed.pos_embed", None)
        sd.pop("vit_pos_embed.pos_embed", None)
        model.load_state_dict(sd, strict=False)
        del sd

        dev = self._device
        model = model.to(dtype=torch.bfloat16, device=dev).eval()
        vae_model = vae_model.to(dev).eval()
        vae_model.requires_grad_(False)

        self.model = model
        self._patch_prepare_outputs_to_device()
        if os.environ.get("PRISM_GRAD_CKPT") == "1":
            self._patch_gradient_checkpointing()
        self._vae = vae_model
        self._tokenizer = tokenizer
        self._new_token_ids = new_token_ids
        self._vae_transform = ImageTransform(1024, 512, 16)
        self._vit_transform = ImageTransform(518, 224, 14)
        self._inferencer = InterleaveInferencer(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            vae_transform=self._vae_transform, vit_transform=self._vit_transform,
            new_token_ids=new_token_ids,
        )
        self._train_ready = True
        return self._inferencer

    def _ensure_packing_cpu(self) -> None:
        if self._new_token_ids is not None:
            return
        _ensure_on_path(self._repo_paths)
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from modeling.qwen2 import Qwen2Tokenizer

        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        self._tokenizer = tokenizer
        self._new_token_ids = new_token_ids
        self._vae_transform = ImageTransform(1024, 512, 16)
        self._vit_transform = ImageTransform(518, 224, 14)

    @staticmethod
    def _to_pil_list(images: Sequence[Any]) -> list:
        from PIL import Image
        out = []
        for im in images:
            out.append(im if isinstance(im, Image.Image) else Image.open(im).convert("RGB"))
        return out

    def _patch_prepare_outputs_to_device(self) -> None:
        if getattr(self.model, "_prism_prepare_device_patch", False):
            return
        import torch

        def move(x):
            if torch.is_tensor(x):
                return x.to(self._device)
            if isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            if isinstance(x, list):
                return [move(v) for v in x]
            if isinstance(x, tuple):
                return tuple(move(v) for v in x)
            return x

        for name in [n for n in dir(self.model) if n.startswith("prepare_")]:
            fn = getattr(self.model, name)
            if not callable(fn):
                continue

            def wrapped(*args, _fn=fn, **kwargs):
                return move(_fn(*args, **kwargs))

            setattr(self.model, name, wrapped)
        self.model._prism_prepare_device_patch = True

    def _patch_gradient_checkpointing(self) -> None:
        import torch
        from torch.utils.checkpoint import checkpoint

        layers = getattr(getattr(self.model.language_model, "model", None), "layers", None)
        if not layers:
            print("[adapter] BAGEL grad-ckpt: no decoder layers found, skipped")
            return
        n = 0
        for layer in layers:
            if getattr(layer, "_prism_grad_ckpt", False):
                continue
            orig = layer.forward

            def fwd(*args, _layer=layer, _orig=orig, **kwargs):
                if not torch.is_grad_enabled():
                    return _orig(*args, **kwargs)
                train_mode = _layer.training
                if "packed_sequence" in kwargs:
                    hidden = kwargs.pop("packed_sequence")
                    call = lambda h: _orig(packed_sequence=h, **kwargs)
                else:
                    hidden, rest = args[0], args[1:]
                    call = lambda h: _orig(h, *rest, **kwargs)

                def run(h):
                    was = _layer.training
                    _layer.train(train_mode)
                    try:
                        return call(h)
                    finally:
                        _layer.train(was)

                return checkpoint(run, hidden, use_reentrant=False)

            layer.forward = fwd
            layer._prism_grad_ckpt = True
            n += 1
        print(f"[adapter] BAGEL grad-ckpt: wrapped {n} decoder layers (use_reentrant=False)")

    def _t2i_kwargs(self, cfg: dict) -> dict:
        h, w = cfg["image_size"][1], cfg["image_size"][0]
        return dict(
            cfg_text_scale=cfg["cfg_scale"],
            cfg_img_scale=cfg["cfg_img_scale"],
            cfg_interval=list(cfg["cfg_interval"]),
            timestep_shift=cfg["timestep_shift"],
            num_timesteps=cfg["num_steps"],
            cfg_renorm_min=cfg["cfg_renorm_min"],
            cfg_renorm_type=cfg["cfg_renorm_type"],
            image_shapes=(h, w),
        )

    def generate_interleaved(
        self, prompt: str, images: Sequence[Any] = (), *, think: bool = True, **gen_cfg: Any
    ) -> Trajectory:
        from ..core.trajectory import from_generation

        self._ensure()
        cfg = {**self._gen_defaults, **gen_cfg}
        max_images = max(1, int(cfg.get("max_images") or 1))
        pil_in = self._to_pil_list(images)
        seed = gen_cfg.get("seed")
        if seed is not None:
            import torch

            torch.manual_seed(int(seed))

        body = _normalize_markers(self._rollout_text(prompt, pil_in, cfg))

        raw, n_markers = _clip_marker_text(body, max_images)
        if n_markers <= 0:
            return from_generation(prompt, raw, images=[],
                                   input_images=pil_in, think_mode=False)

        out_imgs, conditions = self._images_along_text(prompt, body, pil_in, cfg)
        traj = from_generation(prompt, raw, images=out_imgs,
                               input_images=pil_in, think_mode=False)
        traj.meta = {**(getattr(traj, "meta", None) or {}),
                     "image_conditions": conditions}
        return traj

    def _rollout_text(self, prompt: str, cond_images: Sequence[Any], cfg: dict) -> str:
        from data.data_utils import pil_img2rgb

        from ..audit import prompts_bagel

        inf = self._ensure()
        max_images = max(1, int(cfg.get("max_images") or 1))
        with self._unwrapped_lm(), self._autocast():
            ctx = inf.init_gen_context()
            for im in cond_images:
                sized = inf.vae_transform.resize_transform(pil_img2rgb(self._pil(im)))
                ctx = inf.update_context_image(sized, ctx, vae=True, vit=True)
            ctx = inf.update_context_text(
                prompts_bagel.interleave_rollout_prompt(prompt, max_images), ctx
            )
            return inf.gen_text(
                ctx,
                max_length=int(cfg["max_new_tokens"]),
                do_sample=True,
                temperature=cfg["text_temperature"],
            ) or ""

    def _images_along_text(
        self, prompt: str, body: str, cond_images: Sequence[Any], cfg: dict
    ) -> list:
        from copy import deepcopy

        from data.data_utils import pil_img2rgb

        from ..audit import prompts_bagel

        inf = self._ensure()
        max_images = max(1, int(cfg.get("max_images") or 1))
        segments = (body or "").split("<image>")
        n_markers = min(len(segments) - 1, max_images)
        if n_markers <= 0:
            return [], []

        h, w = cfg["image_size"][1], cfg["image_size"][0]
        t2i = self._t2i_kwargs(cfg)
        t2i.pop("image_shapes", None)
        t2i["cfg_img_scale"] = 2.0
        t2i["cfg_interval"] = (0.0, 1.0)

        def _feed(ctx):
            for im in cond_images:
                sized = inf.vae_transform.resize_transform(pil_img2rgb(self._pil(im)))
                ctx = inf.update_context_image(sized, ctx, vae=True, vit=True)
            return ctx

        out_imgs: list = []
        conditions: list[str] = []
        with self._unwrapped_lm(), self._autocast():
            gen_ctx = _feed(inf.init_gen_context())
            cfg_img_ctx = inf.init_gen_context()

            for k in range(n_markers):
                passage = _strip_rollout_scaffold(segments[k])
                if not passage:
                    continue
                conditions.append(passage)
                cfg_text_ctx = deepcopy(gen_ctx)
                gen_ctx = inf.update_context_text(passage, gen_ctx)
                cfg_img_ctx = inf.update_context_text(passage, cfg_img_ctx)
                img = inf.gen_image(
                    (h, w), gen_ctx,
                    cfg_text_precontext=cfg_text_ctx,
                    cfg_img_precontext=cfg_img_ctx,
                    **t2i,
                )
                out_imgs.append(img)
                sized = inf.vae_transform.resize_transform(pil_img2rgb(img))
                gen_ctx = inf.update_context_image(sized, gen_ctx, vae=True, vit=True)
        return out_imgs, conditions

    def _autocast(self):
        import torch

        return torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)

    def regenerate_images(
        self,
        prompt: str,
        gt_text: str,
        images: Sequence[Any] = (),
        *,
        captions: Optional[Sequence[str]] = None,
        **gen_cfg: Any,
    ) -> list[Any]:
        self._ensure()
        cfg = {**self._gen_defaults, **gen_cfg}
        seed = gen_cfg.get("seed")
        if seed is not None:
            import torch

            torch.manual_seed(int(seed))
        body = "".join(f"{c}<image>" for c in captions) if captions else (gt_text or prompt)
        if "<image>" not in body:
            body += "<image>"
        imgs, _ = self._images_along_text(prompt, body, self._to_pil_list(images), cfg)
        return imgs

    def regenerate_images_batch(
        self, prompt: str, n: int, *, image_size=(1024, 1024),
        cfg_scale=4.0, num_steps=50, seed: Optional[int] = None, think: bool = False,
    ) -> list[Any]:
        from copy import deepcopy

        inf = self._ensure()
        cfg = {**self._gen_defaults, "cfg_scale": cfg_scale, "num_steps": num_steps,
               "image_size": image_size}
        if seed is not None:
            import torch

            torch.manual_seed(int(seed))
        h, w = cfg["image_size"][1], cfg["image_size"][0]
        t2i = self._t2i_kwargs(cfg)
        t2i.pop("image_shapes", None)
        with self._unwrapped_lm(), self._autocast():
            return inf.gen_images_batch(prompt, (h, w), n, **t2i)

    def understand(
        self, images: Sequence[Any], question: str, *, history: Optional[list] = None, **gen_cfg: Any
    ) -> str:
        inf = self._ensure()
        cfg = {**self._gen_defaults, **gen_cfg}
        think = bool(gen_cfg.get("think", False))
        pil = self._to_pil_list(images)
        with self._unwrapped_lm():
            outputs = inf.interleave_inference(
                input_lists=[*pil, question] if pil else [question],
                understanding_output=True,
                think=think,
                do_sample=False,
                text_temperature=cfg["text_temperature"],
                max_think_token_n=int(cfg["max_new_tokens"]),
            )
        for x in outputs:
            if isinstance(x, str):
                return x
        return ""

    def regenerate_text(
        self, prompt: str, images: Sequence[Any], *, grounded_instruction: str = "", **gen_cfg: Any
    ) -> str:
        q = (grounded_instruction + "\n" + prompt) if grounded_instruction else prompt
        return self.understand(images, q, **gen_cfg)

    def _packer(self):
        if self._pack_shim is not None:
            return self._pack_shim
        from types import SimpleNamespace

        from data.data_utils import get_flattened_position_ids_extrapolate
        from data.dataset_base import DataConfig

        self._pack_shim = SimpleNamespace(
            data_config=DataConfig(
                grouped_datasets=None,
                text_cond_dropout_prob=0.0,
                vit_cond_dropout_prob=0.0,
                vae_cond_dropout_prob=0.0,
                vae_image_downsample=16,
                max_latent_size=64,
            ),
            get_flattened_position_ids=get_flattened_position_ids_extrapolate,
            use_flex=True,
            max_num_tokens=81920,
            **self._new_token_ids,
        )
        return self._pack_shim

    def _new_builder(self) -> _PlanBuilder:
        return _PlanBuilder(self._tokenizer)

    def _pack(self, builders: Sequence[_PlanBuilder]) -> tuple[dict, dict]:
        from data.dataset_base import PackedDataset

        shim = self._packer()
        status = PackedDataset.set_sequence_status(shim)
        tags = {"ce": [], "mse": []}
        for b in builders:
            status = PackedDataset.pack_sequence(shim, b.sample(), status)
            tags["ce"] += b.ce_tags
            tags["mse"] += b.mse_tags
        return PackedDataset.to_tensor(shim, status), tags

    def _dummy_builder(self, *, need_vit: bool, need_gen: bool) -> _PlanBuilder:
        import torch

        b = self._new_builder()
        if need_vit:
            b.vit(torch.zeros(3, 28, 28))
        if need_gen:
            b.vae(torch.zeros(3, 16, 16), loss=1, tag="dummy")
        return b

    def _pil(self, x):
        from PIL import Image

        im = x if hasattr(x, "convert") else Image.open(x)
        return im.convert("RGB") if im.mode != "RGB" else im

    def _vae_tensor(self, im):
        return self._vae_transform(self._pil(im))

    def _vit_tensor(self, im):
        return self._vit_transform(self._pil(im))

    def _teach_builders(
        self,
        sample: Any,
        *,
        drop_text: bool = False,
        image_override: Optional[Sequence[Any]] = None,
        body_override: Optional[str] = None,
    ) -> list[_PlanBuilder]:
        kind = sample.kind
        steps = list(sample.steps or [])
        if not steps:
            if kind == "t2i":
                imgs = sample.target_image if isinstance(sample.target_image, list) else (
                    [sample.target_image] if sample.target_image is not None else [])
                base = sample.context_text or ""
                steps = [InterleaveStep(text=(base if i == 0 else ""), image=im)
                         for i, im in enumerate(imgs)]
            elif kind in ("i2t", "s2t"):
                steps = [InterleaveStep(text="", image=im) for im in (sample.cond_images or [])]
                steps.append(InterleaveStep(text=sample.target_text or "", image=None))
            elif kind == "neg":
                steps = [InterleaveStep(text=sample.neg_text or "", image=None)]
        if body_override is not None:
            steps = [InterleaveStep(text="", image=st.image) for st in steps if st.image is not None]
            steps.append(InterleaveStep(text=body_override, image=None))

        text_loss = 1 if kind in ("t2i", "i2t", "s2t", "neg") else 0
        text_tag = "negk" if kind == "neg" else "keep"

        b = self._new_builder()
        user_imgs = list(sample.cond_images or [])
        if kind in ("i2t", "s2t") and not (sample.steps or []):
            user_imgs = []
        if not drop_text and sample.prompt:
            b.text(_bagel_text(sample.prompt), loss=0)
        for im in user_imgs:
            b.vit(self._vit_tensor(im))

        gi = 0
        for idx, st in enumerate(steps):
            if st.text and not drop_text:
                b.text(_bagel_text(st.text), loss=text_loss, tag=text_tag,
                       image_marker=(kind in ("t2i", "i2t")
                                     and text_loss == 1 and text_tag == "keep"
                                     and st.image is not None))
            if st.image is None:
                continue
            if kind in ("t2i", "i2t"):
                img = st.image
                if image_override is not None and gi < len(image_override):
                    img = image_override[gi]
                gi += 1
                t = self._vae_tensor(img)
                b.vae(t, loss=1, tag="real")
                if idx < len(steps) - 1:
                    b.vae(t.clone(), loss=0)
                    b.vit(self._vit_tensor(img))
            else:
                b.vit(self._vit_tensor(st.image))

        builders = [b]
        if not b.has_vit or not b.has_gen:
            builders.append(self._dummy_builder(need_vit=not b.has_vit, need_gen=not b.has_gen))
        return builders

    def _mixed_forward(self, data: dict) -> dict:
        import torch

        model = self.model
        dev = next(model.parameters()).device
        data = dict(data)
        data.pop("batch_data_indexes", None)
        data.pop("ce_loss_weights", None)
        for k, v in list(data.items()):
            if torch.is_tensor(v):
                data[k] = v.to(dev)
        if "nested_attention_masks" in data:
            data["nested_attention_masks"] = [m.to(dev, torch.bfloat16)
                                              for m in data["nested_attention_masks"]]

        was = model.training
        model.train()
        try:
            with self._unwrapped_lm(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                padded = data.pop("padded_images", None)
                if padded is not None:
                    with torch.no_grad():
                        latents = [self._vae.encode(padded[i:i + 4].to(dev))
                                   for i in range(0, len(padded), 4)]
                        data["padded_latent"] = torch.cat(latents, dim=0)
                out = model(**data)
        finally:
            model.train(was)
        return out

    def _fm_reduce(self, mse, tags: dict):
        per_tok = mse.float().mean(dim=-1)
        off, n_real = 0, 0
        w = per_tok.new_zeros(per_tok.shape)
        for tag, n in tags["mse"]:
            if tag == "real":
                w[off:off + n] = 1.0
                n_real += n
            off += n
        assert off == per_tok.numel(), f"mse rows {per_tok.numel()} != tagged {off}"
        if n_real:
            return (per_tok * w).sum() / n_real
        return per_tok.sum() * 0.0

    def _ce_reduce(self, ce, tags: dict):
        import torch

        zero = next(self.model.parameters()).new_zeros(()).float()
        if ce is None:
            return zero, zero
        ce = ce.float()
        off = 0
        keep_rows, neg_rows = [], []
        for tag, n in tags["ce"]:
            rows = ce[off:off + n]
            (keep_rows if tag == "keep" else neg_rows).append(
                rows if tag == "keep" else rows[:-1] if n > 1 else rows)
            off += n
        assert off == ce.numel(), f"ce rows {ce.numel()} != tagged {off}"
        ce_loss = torch.cat(keep_rows).mean() if keep_rows else zero
        neg = zero
        if neg_rows:
            tok_logp = -torch.cat(neg_rows)
            p = tok_logp.exp().clamp(max=float(self._train_cfg["unlike_clamp"]))
            neg = -(1.0 - p).log().mean()
        return ce_loss, neg

    def interleaved_teach_loss(self, sample: Any, *, objective: str = "sft") -> dict:
        if objective != "sft":
            raise NotImplementedError("only the standard SFT objective is supported")
        import torch

        from ..training.objective import quadrant_loss_weights

        self._ensure_trainable()
        drop = False
        if sample.kind == "t2i":
            p = float(self._train_cfg["cfg_txt_drop_prob"])
            drop = p > 0 and bool(torch.rand(()).item() < p)

        data, tags = self._pack(self._teach_builders(sample, drop_text=drop))
        out = self._mixed_forward(data)
        fm = self._fm_reduce(out["mse"], tags)
        ce, neg = self._ce_reduce(out["ce"], tags)
        w = quadrant_loss_weights(sample.kind)
        loss = (w["fm"] * float(self._train_cfg["mse_weight"]) * fm
                + w["ce"] * float(self._train_cfg["ce_weight"]) * ce
                + w["neg"] * neg)
        return {"loss": loss, "fm": fm.detach(), "ce": ce.detach(), "neg": neg.detach(),
                "kind": sample.kind, "cfg_drop": drop}

    def trainable_modules(self) -> dict[str, list[str]]:
        return {
            "und": ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
                    "gate_proj", "up_proj", "down_proj",
                    "input_layernorm", "post_attention_layernorm", "model.norm."],
            "gen": ["_moe_gen"],
        }

    @contextmanager
    def _unwrapped_lm(self) -> Iterator[None]:
        lm = getattr(self.model, "language_model", None)
        inner = getattr(lm, "module", None)
        if inner is None or inner is lm:
            yield
            return
        self.model.language_model = inner
        try:
            yield
        finally:
            self.model.language_model = lm

    def _lora_layers(self) -> list:
        try:
            from peft.tuners.tuners_utils import BaseTunerLayer
        except Exception:
            return []
        if self.model is None:
            return []
        return [m for m in self.model.modules() if isinstance(m, BaseTunerLayer)]

    @contextmanager
    def _lora_disabled(self) -> Iterator[None]:
        layers = self._lora_layers()
        if not layers:
            yield
            return
        prev = [getattr(m, "_disable_adapters", False) for m in layers]
        for m in layers:
            m._disable_adapters = True
        try:
            yield
        finally:
            for m, p in zip(layers, prev):
                m._disable_adapters = p

    @contextmanager
    def frozen_base_context(self) -> Iterator[None]:
        layers = self._lora_layers()
        if not layers:
            if not getattr(self, "_warned_no_peft", False):
                self._warned_no_peft = True
                print("[adapter] BAGEL frozen_base_context: no LoRA layer to disable, "
                      "auditing and teaching targets will use the current weights")
            with self._unwrapped_lm():
                yield
            return
        if not getattr(self, "_logged_peft_ok", False):
            self._logged_peft_ok = True
            print(f"[adapter] BAGEL frozen_base_context: disabling {len(layers)} LoRA "
                  "layers (frozen base)")
        with self._lora_disabled(), self._unwrapped_lm():
            yield

    @contextmanager
    def unwrapped_context(self) -> Iterator[None]:
        with self._unwrapped_lm():
            yield

    @contextmanager
    def generation_mode(self) -> Iterator[None]:
        self._ensure()
        model = self.model
        was = model.training
        model.eval()
        try:
            yield
        finally:
            model.train(was)

    def save_adapter(self, path: str) -> None:
        raise NotImplementedError(_STUB.format(what="save_adapter"))

    def load_adapter(self, path: str) -> None:
        raise NotImplementedError(_STUB.format(what="load_adapter"))

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise NotImplementedError(_STUB.format(what="tokenizer (backend not loaded)"))
        return self._tokenizer

    @property
    def gen_defaults(self) -> dict:
        return dict(self._gen_defaults)
