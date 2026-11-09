
from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

from ..core.trajectory import from_generation
from ..core.types import Trajectory
from .base import UMMAdapter
from .registry import register

_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a "
    "<think></think> block and place all reasoning inside it. You MUST interleave text with "
    "generated images using tags like <image1>, <image2>. Images can ONLY be generated between "
    "<think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no "
    "reasoning is needed, directly provide the answer without reasoning. Do not use tags like "
    "<image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, "
    "always provide a concise, user-facing final answer. The answer may include text, images, or "
    "both. Match the user's language in both reasoning and the final answer."
)


def _ensure_on_path(repo_paths: Optional[dict[str, str]] = None) -> None:
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundled_src = os.path.join(pkg_dir, "models", "SenseNova-U1", "src")
    if os.path.isdir(os.path.join(bundled_src, "sensenova_u1")):
        if bundled_src not in sys.path:
            sys.path.insert(0, bundled_src)
        return

    su1 = (repo_paths or {}).get("sensenova_u1")
    if su1:
        for p in (os.path.join(su1, "src"), os.path.join(su1, "training", "flow-grpo")):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)


def build_isolation_attend_mask(
    document_ids, modality_indicators, image_gen_indicators, token_pos, dup_boundary
):
    import torch

    L = int(document_ids.numel())
    tp_q = token_pos.view(L, 1); tp_k = token_pos.view(1, L)
    doc_q = document_ids.view(L, 1); doc_k = document_ids.view(1, L)
    mod_q = modality_indicators.view(L, 1); mod_k = modality_indicators.view(1, L)
    gen_q = image_gen_indicators.view(L, 1).bool(); gen_k = image_gen_indicators.view(1, L).bool()
    dup_k = dup_boundary.view(1, L).bool()

    same_doc = doc_q == doc_k
    causal = tp_q >= tp_k
    same_img = (mod_q > 0) & (mod_q == mod_k) & same_doc
    base = (causal & same_doc) | same_img

    same_img_qk = (mod_q == mod_k) & same_doc
    gate1 = (~gen_k) | (gen_q & same_img_qk)
    gate2 = (~gen_q) | gen_k | (tp_k != tp_q)
    gen_gate = gate1 & gate2

    return base & gen_gate & (~dup_k)


def build_isolation_block_mask(
    document_ids, modality_indicators, image_gen_indicators, token_pos, dup_boundary
):
    from torch.nn.attention.flex_attention import and_masks, create_block_mask, or_masks

    L = int(document_ids.numel())
    dev = document_ids.device
    gen = image_gen_indicators.bool()
    dup = dup_boundary.bool()

    def causal(b, h, q, kv):
        return token_pos[q] >= token_pos[kv]

    def samedoc(b, h, q, kv):
        return document_ids[q] == document_ids[kv]

    def sameimg(b, h, q, kv):
        return (modality_indicators[q] > 0) & (modality_indicators[q] == modality_indicators[kv]) \
            & (document_ids[q] == document_ids[kv])

    def gen_gate(b, h, q, kv):
        same_img = (modality_indicators[q] == modality_indicators[kv]) & (document_ids[q] == document_ids[kv])
        gate1 = (~gen[kv]) | (gen[q] & same_img)
        gate2 = (~gen[q]) | gen[kv] | (token_pos[kv] != token_pos[q])
        return gate1 & gate2

    def dup_gate(b, h, q, kv):
        return ~dup[kv]

    mask_mod = and_masks(or_masks(and_masks(causal, samedoc), sameimg), gen_gate, dup_gate)
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=dev, _compile=False)


def _flags_to_indicators(modality_indicators, per_image_flags, image_id_base: int = 1):
    import torch

    out = torch.zeros_like(modality_indicators, dtype=torch.bool)
    img_mask = modality_indicators >= 0
    if not bool(img_mask.any()):
        return out
    flags = torch.as_tensor(list(per_image_flags), dtype=torch.bool, device=modality_indicators.device)
    img_ids = modality_indicators[img_mask] - image_id_base
    valid = (img_ids >= 0) & (img_ids < flags.numel())
    if bool(valid.any()):
        out_idx = img_mask.nonzero(as_tuple=True)[0][valid]
        out[out_idx] = flags[img_ids[valid]]
    return out


def _build_dup_boundary(modality_indicators, gen_tok, dup_tok):
    dup_boundary = gen_tok.new_zeros(gen_tok.shape, dtype=gen_tok.dtype)
    boundary = modality_indicators == -1
    dup_boundary[:-1] |= boundary[:-1] & dup_tok[1:]
    dup_boundary[1:-2] |= boundary[1:-2] & gen_tok[:-3] & boundary[2:-1] & dup_tok[3:]
    return dup_boundary


def build_interleaved_indicators(
    input_ids, img_start_id: int, img_context_id: int, image_for_gen_flags, is_dup_flags,
    img_end_id: Optional[int] = None,
):
    import torch

    ids = input_ids.view(-1)
    L = ids.numel()
    device = ids.device
    is_start = (ids == img_start_id).long()
    shifted = torch.cat([torch.zeros(1, dtype=torch.long, device=device), is_start])[:-1]
    modality = shifted.cumsum(0)
    modality[ids != img_context_id] = -1

    gen = _flags_to_indicators(modality, image_for_gen_flags)
    dup = _flags_to_indicators(modality, is_dup_flags)
    dup_boundary = _build_dup_boundary(modality, gen, dup)

    not_img = (ids != img_context_id).long()
    token_pos = (shifted + not_img).cumsum(0) - 1
    if bool(dup.any()):
        starts = (ids == img_start_id).nonzero(as_tuple=True)[0]
        ends = (ids == img_end_id).nonzero(as_tuple=True)[0] if img_end_id is not None else None
        for d in torch.unique(modality[dup]).tolist():
            assert d >= 2, "dup image cannot be the first image"
            s0, s1 = int(starts[d - 2]), int(starts[d - 1])
            e0 = int(ends[d - 2]) if ends is not None else s1 - 1
            e1 = int(ends[d - 1]) if ends is not None else None
            if e1 is None:
                cur = (modality == d).nonzero(as_tuple=True)[0]
                e1 = int(cur[-1]) + 1
            assert e1 - s1 == e0 - s0, "dup span must be adjacent and same length as its source"
            diff = int(token_pos[e1] - token_pos[e0])
            token_pos[s1 : e1 + 1] = token_pos[s0 : e0 + 1]
            if e1 + 1 < token_pos.numel():
                token_pos[e1 + 1 :] -= diff

    document_ids = torch.zeros(L, dtype=torch.long, device=device)
    return {
        "modality_indicators": modality,
        "image_gen_indicators": gen,
        "image_dup_indicators": dup,
        "dup_boundary": dup_boundary,
        "token_pos": token_pos,
        "document_ids": document_ids,
    }


@register("sensenova_u1")
def _build(**kwargs: Any) -> "SenseNovaU1Adapter":
    return SenseNovaU1Adapter(**kwargs)


class SenseNovaU1Adapter(UMMAdapter):
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        repo_paths: Optional[dict[str, str]] = None,
        gen_defaults: Optional[dict] = None,
        attn_backend: str = "auto",
        fm_params: Optional[dict] = None,
        und_context_max_pixels: Optional[int] = None,
    ) -> None:
        _ensure_on_path(repo_paths)
        import torch
        import sensenova_u1
        from sensenova_u1.utils import load_model_and_tokenizer

        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        import warnings
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")

        sensenova_u1.set_attn_backend(attn_backend)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self._device = device
        self._torch = torch
        self.model, self._tokenizer = load_model_and_tokenizer(
            model_path, dtype=torch_dtype, device=device
        )
        self._gen_defaults = {
            "cfg_scale": 4.0,
            "img_cfg_scale": 1.0,
            "timestep_shift": 3.0,
            "cfg_interval": (0.0, 1.0),
            "num_steps": 50,
            "image_size": (1024, 1024),
            "system_message": DEFAULT_SYSTEM_MESSAGE,
            "seed": 42,
            "max_new_tokens": 8192,
            **(gen_defaults or {}),
        }
        self._neo_inferencer = None
        self._fm = {"P_mean": -0.8, "P_std": 0.8, "timestep_shift": 1.0, "t_eps": 0.05, **(fm_params or {})}
        self._und_context_max_pixels = und_context_max_pixels
        import os as _os
        self._attn_impl = _os.environ.get("PRISM_ATTN", "sdpa")
        self._attn_impl = self._validate_attn_impl(self._attn_impl, "PRISM_ATTN")
        env_rollout = _os.environ.get("PRISM_ROLLOUT_ATTN")
        if env_rollout:
            self._rollout_attn_impl = self._validate_attn_impl(env_rollout, "PRISM_ROLLOUT_ATTN")
        else:
            try:
                self._rollout_attn_impl = self._causal_lm().model.layers[0].self_attn.config._attn_implementation
            except Exception:
                self._rollout_attn_impl = None

    @staticmethod
    def _validate_attn_impl(impl: str, who: str) -> str:
        if impl == "eager":
            return impl
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except Exception:
            return impl
        if impl not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(
                f"{who}={impl!r} 不是可用的注意力实现; 可选: "
                f"eager, {', '.join(sorted(ALL_ATTENTION_FUNCTIONS))}"
            )
        return impl

    def _to_pil(self, batch: Any):
        import numpy as np
        from PIL import Image

        t = self._torch
        mean = t.tensor(_NORM_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
        std = t.tensor(_NORM_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
        arr = (batch.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        arr = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr[0])

    def _g(self, gen_cfg: dict, *keys: str) -> dict:
        merged = {**self._gen_defaults, **gen_cfg}
        return {k: merged[k] for k in keys if k in merged}

    def generate_interleaved(
        self,
        prompt: str,
        images: Sequence[Any] = (),
        *,
        think: bool = True,
        **gen_cfg: Any,
    ) -> Trajectory:
        cfg = self._g(
            gen_cfg,
            "cfg_scale",
            "img_cfg_scale",
            "timestep_shift",
            "cfg_interval",
            "num_steps",
            "image_size",
            "system_message",
            "seed",
            "max_images",
        )
        from types import SimpleNamespace

        mnt = {**self._gen_defaults, **gen_cfg}.get("max_new_tokens")
        gen_config = SimpleNamespace(max_new_tokens=mnt) if mnt else None
        with self._torch.inference_mode():
            text, image_tensors = self.model.interleave_gen(
                self._tokenizer,
                prompt,
                images=list(images),
                generation_config=gen_config,
                think_mode=think,
                verbose=False,
                **cfg,
            )
        pil_images = [self._to_pil(t) for t in image_tensors]
        return from_generation(prompt, text, images=pil_images,
                               input_images=list(images), think_mode=think)

    def regenerate_images(
        self,
        prompt: str,
        gt_text: str,
        images: Sequence[Any] = (),
        *,
        captions: Optional[Sequence[str]] = None,
        **gen_cfg: Any,
    ) -> list[Any]:
        cfg = self._g(
            gen_cfg,
            "cfg_scale",
            "img_cfg_scale",
            "timestep_shift",
            "cfg_interval",
            "num_steps",
            "image_size",
            "max_images",
        )
        seed = gen_cfg.get("seed", None)
        if seed is not None:
            self._torch.manual_seed(int(seed))
        with self._torch.inference_mode():
            image_tensors = self.model.interleave_gen_image_only(
                self._tokenizer,
                prompt,
                gt_text,
                images=list(images),
                **cfg,
            )
        return [self._to_pil(t) for t in image_tensors]

    def understand(
        self,
        images: Sequence[Any],
        question: str,
        *,
        history: Optional[list] = None,
        think: bool = True,
        **gen_cfg: Any,
    ) -> str:
        images = list(images)
        gc = {**self._gen_defaults, **gen_cfg}
        generation_config = dict(max_new_tokens=gc["max_new_tokens"], do_sample=False)
        if gc.get("do_sample"):
            generation_config.update(
                do_sample=True,
                temperature=gc.get("temperature", 0.6),
                top_p=gc.get("top_p", 0.95),
                top_k=gc.get("top_k", 20),
            )
        if gc.get("repetition_penalty"):
            generation_config["repetition_penalty"] = gc["repetition_penalty"]

        if not images:
            return self._text_only(question, generation_config, think=think)

        from sensenova_u1.models.neo_unify.utils import load_image_native

        pvs, ghws = [], []
        for im in images:
            pv, ghw = load_image_native(im)
            pvs.append(pv.to(self._device, dtype=self.model.dtype))
            ghws.append(ghw.to(self._device))
        pixel_values = self._torch.cat(pvs)
        grid_hw = self._torch.cat(ghws)
        if think:
            if question.count("<image>") < len(images):
                question = "<image>\n" * len(images) + question
            with self._torch.inference_mode():
                response, _ = self.model.chat(
                    self._tokenizer, pixel_values, question, generation_config,
                    history=history, return_history=True, grid_hw=grid_hw,
                )
            return response
        from sensenova_u1.models.neo_unify.conversation import get_conv_template

        tok = self._tokenizer
        IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"
        self.model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT)
        self.model.img_start_token_id = tok.convert_tokens_to_ids(IMG_START)
        if question.count("<image>") < len(images):
            question = "<image>\n" * len(images) + question
        conv = get_conv_template(self.model.template)
        conv.system_message = self.model.system_message
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt() + "<think>\n\n</think>\n\n"
        for i in range(grid_hw.shape[0]):
            num = int(grid_hw[i, 0] * grid_hw[i, 1] * self.model.downsample_ratio ** 2)
            query = query.replace("<image>", IMG_START + IMG_CONTEXT * num + IMG_END, 1)
        inputs = tok(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        attn = inputs["attention_mask"].to(self._device)
        gc = dict(generation_config)
        gc.setdefault("eos_token_id", tok.convert_tokens_to_ids(conv.sep.strip()))
        with self._torch.inference_mode():
            out = self.model.generate(
                pixel_values=pixel_values, input_ids=input_ids, grid_hw=grid_hw, attention_mask=attn, **gc
            )
        text = tok.batch_decode(out, skip_special_tokens=True)[0]
        return text.split(conv.sep.strip())[0].strip()

    def _text_only(self, question: str, generation_config: dict, *, think: bool = False,
                   prefill: Optional[str] = None) -> str:
        from sensenova_u1.models.neo_unify.conversation import get_conv_template

        tok = self._tokenizer
        self.model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_start_token_id = tok.convert_tokens_to_ids("<img>")

        conv = get_conv_template(self.model.template)
        conv.system_message = self.model.system_message
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        if prefill is not None:
            query += prefill
        elif not think:
            query += "<think>\n\n</think>\n\n"

        inputs = tok(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        attn = inputs["attention_mask"].to(self._device)
        gc = dict(generation_config)
        gc.setdefault("eos_token_id", tok.convert_tokens_to_ids(conv.sep.strip()))
        with self._torch.inference_mode():
            out = self.model.generate(
                pixel_values=None, input_ids=input_ids, grid_hw=None, attention_mask=attn, **gc
            )
        if prefill is not None:
            new = out[0, input_ids.shape[1]:]
            return tok.decode(new, skip_special_tokens=True).split(conv.sep.strip())[0].strip()
        text = tok.batch_decode(out, skip_special_tokens=True)[0]
        return text.split(conv.sep.strip())[0].strip()

    def regenerate_text(
        self,
        prompt: str,
        images: Sequence[Any],
        *,
        grounded_instruction: str = "",
        **gen_cfg: Any,
    ) -> str:
        q = prompt if not grounded_instruction else f"{prompt}\n\n{grounded_instruction}"
        return self.understand(images, q, **gen_cfg)

    def _require_inferencer(self):
        if self._neo_inferencer is None:
            from flow_grpo.neounify import NeoUnifyInferencer

            self._neo_inferencer = NeoUnifyInferencer(self.model, self._tokenizer)
        return self._neo_inferencer

    def _lm_module(self):
        lm = self.model.language_model
        return getattr(lm, "module", lm)

    @contextmanager
    def _unwrapped_language_model_context(self) -> Iterator[None]:
        lm = self.model.language_model
        raw = self._causal_lm()
        if raw is lm:
            yield
            return
        self.model.language_model = raw
        try:
            yield
        finally:
            self.model.language_model = lm

    @contextmanager
    def generation_mode(self):
        if getattr(self, "_in_gen_mode", False):
            yield
            return
        self._set_attn_impl(getattr(self, "_rollout_attn_impl", None))
        lm = self.model.language_model
        peft = None
        cur, seen = lm, set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if hasattr(cur, "merge_adapter") and hasattr(cur, "get_base_model"):
                peft = cur
                break
            cur = getattr(cur, "module", None)
        if peft is None:
            self._in_gen_mode = True
            try:
                yield
            finally:
                self._in_gen_mode = False
            return
        self._in_gen_mode = True
        peft.merge_adapter()
        self.model.language_model = peft.get_base_model()
        try:
            yield
        finally:
            self.model.language_model = lm
            peft.unmerge_adapter()
            self._in_gen_mode = False

    def _und_teacher_forced(self, prompt: str, cond_images: Sequence[Any], target_text: str):
        torch = self._torch
        model = self.model
        tok = self._tokenizer
        from sensenova_u1.models.neo_unify.conversation import get_conv_template
        from sensenova_u1.models.neo_unify.utils import load_image_native

        IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"
        model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT)
        model.img_start_token_id = tok.convert_tokens_to_ids(IMG_START)

        cond_images = list(cond_images)
        pixel_values, grid_hw = None, None
        if cond_images:
            pvs, ghws = [], []
            for im in cond_images:
                pv, ghw = load_image_native(im)
                pvs.append(pv.to(self._device, dtype=model.dtype))
                ghws.append(ghw.to(self._device))
            pixel_values, grid_hw = torch.cat(pvs), torch.cat(ghws)

        question = ("<image>\n" * len(cond_images)) + prompt

        def _compose(answer):
            t = get_conv_template(model.template)
            t.system_message = model.system_message
            t.append_message(t.roles[0], question)
            t.append_message(t.roles[1], answer)
            q = t.get_prompt()
            if grid_hw is not None:
                for i in range(grid_hw.shape[0]):
                    num = int(grid_hw[i, 0] * grid_hw[i, 1] * model.downsample_ratio**2)
                    q = q.replace("<image>", IMG_START + IMG_CONTEXT * num + IMG_END, 1)
            return q

        ids_ctx = tok(_compose(None), return_tensors="pt").input_ids
        ids_full = tok(_compose(target_text), return_tensors="pt").input_ids.to(self._device)
        ctx_len = ids_ctx.shape[1]

        labels = ids_full.clone()
        labels[:, :ctx_len] = -100

        indexes = model.get_thw_indexes(ids_full[0], grid_hw)
        emb = self._lm_module().get_input_embeddings()(ids_full)
        if pixel_values is not None:
            vit = model.extract_feature(pixel_values, grid_hw=grid_hw)
            B, N, C = emb.shape
            emb = emb.reshape(B * N, C)
            sel = ids_full.reshape(B * N) == model.img_context_token_id
            emb[sel] = vit.reshape(-1, C).to(emb.dtype)
            emb = emb.reshape(B, N, C)
        attn = torch.ones_like(ids_full)
        return emb, indexes, attn, labels, ids_full

    def _fm_noise_scale(self, grid_h: int, grid_w: int) -> float:
        model = self.model
        merge = int(round(1 / model.downsample_ratio))
        ns = float(model.noise_scale)
        if model.noise_scale_mode in ("resolution", "dynamic"):
            merged = (grid_h * grid_w) // (merge ** 2)
            ns = (merged / float(model.noise_scale_base_image_seq_len)) ** 0.5 * float(model.noise_scale)
        return ns

    def _encode_gen_x(self, pil):
        import numpy as np

        torch = self._torch
        model = self.model
        inf = self._require_inferencer()
        pm = inf.patch_size * inf.merge_size
        pil = pil.convert("RGB")
        W, H = pil.size
        if W % pm or H % pm:
            raise ValueError(f"image {W}x{H} not divisible by patch*merge={pm}")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        x_img = (torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device, dtype=model.dtype) * 2.0) - 1.0
        x = model.patchify(x_img, pm)
        seq_len = (H // pm) * (W // pm)
        noise_scale = self._fm_noise_scale(H // inf.patch_size, W // inf.patch_size)
        return x, (W, H), seq_len, noise_scale

    def _causal_lm(self):
        lm = self.model.language_model
        lm = getattr(lm, "module", lm)
        if hasattr(lm, "get_base_model"):
            lm = lm.get_base_model()
        return lm

    def _set_attn_impl(self, impl) -> None:
        if not impl:
            return
        for layer in self._causal_lm().model.layers:
            layer.self_attn.config._attn_implementation = impl

    def _ensure_eager(self):
        self._set_attn_impl(getattr(self, "_attn_impl", "sdpa"))

    def _gen_image_tokens(self, pil, shared=None):
        torch = self._torch
        model = self.model
        inf = self._require_inferencer()
        ps, pm = inf.patch_size, inf.patch_size * inf.merge_size
        x, (W, H), seq_len, ns = self._encode_gen_x(pil)
        gh, gw = H // ps, W // ps
        P_mean, P_std, shift = self._fm["P_mean"], self._fm["P_std"], self._fm["timestep_shift"]
        t_eps = self._fm.get("t_eps", getattr(model.config, "t_eps", 0.05))
        if shared is not None:
            u, eps = shared["u"], shared["eps"]
            if eps.shape != x.shape:
                raise ValueError(
                    f"shared-noise shape {tuple(eps.shape)} != image {tuple(x.shape)} — "
                    "shared-noise image targets must share resolution"
                )
        else:
            eps = torch.randn_like(x) * ns
            u = torch.randn((), device=self._device, dtype=torch.float32) * P_std + P_mean
        t = model._apply_time_schedule(torch.sigmoid(u).to(x.dtype), seq_len, shift)
        z = t * x + (1.0 - t) * eps
        v = (x - z) / (1.0 - t).clamp_min(t_eps)
        img_pred = model.unpatchify(z, pm, H, W)
        img_in = model.patchify(img_pred, ps, channel_first=True)
        embeds = model.extract_feature(
            img_in.view(gh * gw, -1).to(model.dtype), gen_model=True,
            grid_hw=torch.tensor([[gh, gw]], device=self._device),
        ).view(1, seq_len, -1)
        te = model.fm_modules["timestep_embedder"](t.expand(seq_len)).view(1, seq_len, -1)
        if model.add_noise_scale_embedding:
            te = te + model.fm_modules["noise_scale_embedder"](
                torch.full_like(t.expand(seq_len), float(ns) / model.noise_scale_max_value)
            ).view(1, seq_len, -1)
        return {"embeds": embeds + te, "z": z, "v": v, "t": t, "u": u, "eps": eps,
                "seq_len": seq_len, "grid": (gh, gw), "t_eps": t_eps}

    def _und_context_tokens(self, pil, n_images: int = 1):
        from PIL import Image
        from sensenova_u1.models.neo_unify.utils import load_image_native

        model = self.model
        max_pixels = self._und_context_max_pixels
        if max_pixels is None:
            max_pixels = min(2048 * 2048, (4096 * 4096) // max(1, n_images))
        pv, ghw = load_image_native(
            pil if hasattr(pil, "convert") else Image.open(pil),
            model.patch_size,
            model.downsample_ratio,
            min_pixels=512 * 512,
            max_pixels=max_pixels,
            upscale=False,
        )
        pv = pv.to(self._device, dtype=model.dtype)
        ghw = ghw.to(self._device)
        embeds = model.extract_feature(pv, grid_hw=ghw)
        gh, gw = int(ghw[0, 0]), int(ghw[0, 1])
        n_ctx = int(gh * gw * (model.downsample_ratio ** 2))
        return embeds.view(n_ctx, -1), n_ctx, (gh, gw)

    def _und_embeds_native(self, pil):
        import numpy as np

        torch = self._torch
        model = self.model
        inf = self._require_inferencer()
        ps = inf.patch_size
        pil = pil.convert("RGB")
        W, H = pil.size
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).to(self._device, dtype=model.dtype)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device, dtype=model.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self._device, dtype=model.dtype).view(3, 1, 1)
        t = (t - mean) / std
        gh, gw = H // ps, W // ps
        pv = t.view(3, gh, ps, gw, ps).permute(1, 3, 0, 2, 4).reshape(gh * gw, 3 * ps * ps)
        emb = model.extract_feature(pv, grid_hw=torch.tensor([[gh, gw]], device=self._device))
        n_ctx = int(gh * gw * (model.downsample_ratio ** 2))
        return emb.view(n_ctx, -1), n_ctx, (gh, gw)

    def _teach_forward(self, sample, *, body_override=None, image_override=None,
                       shared_noise=None, capture_noise=None, drop=""):
        from PIL import Image
        from sensenova_u1.models.neo_unify.conversation import get_conv_template

        from ..core.types import InterleaveStep

        torch = self._torch
        F = torch.nn.functional
        model = self.model
        tok = self._tokenizer
        dev, dt = self._device, model.dtype
        self._ensure_eager()
        IMG_START = tok.convert_tokens_to_ids("<img>")
        IMG_CTX = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        IMG_END = tok.convert_tokens_to_ids("</img>")
        model.img_start_token_id, model.img_context_token_id = IMG_START, IMG_CTX
        inf = self._require_inferencer()
        pm = inf.patch_size * inf.merge_size
        kind = sample.kind

        steps = list(sample.steps or [])
        if not steps:
            if kind == "t2i":
                imgs = sample.target_image if isinstance(sample.target_image, list) else (
                    [sample.target_image] if sample.target_image is not None else [])
                base = sample.context_text or ""
                steps = [InterleaveStep(text=(base if i == 0 else ""), image=im) for i, im in enumerate(imgs)]
            elif kind == "i2t":
                imgs = list(sample.cond_images or [])
                steps = [InterleaveStep(text="", image=im) for im in imgs]
                steps.append(InterleaveStep(text=sample.target_text or "", image=None))
            elif kind == "neg":
                steps = [InterleaveStep(text=sample.neg_text or "", image=None)]

        if drop == "txtimg":
            gen_pils = [st.image for st in steps if st.image is not None]
            if gen_pils:
                pick = int(torch.randint(len(gen_pils), ()).item())
                steps = [InterleaveStep(text="", image=gen_pils[pick])]
            else:
                drop = "txt"

        def _pil(x):
            return x if hasattr(x, "convert") else Image.open(x)

        img_embeds: list = []
        gen_flags: list[int] = []
        dup_flags: list[int] = []
        grids: list = []
        fm_list: list = []

        def _reg(embeds, n_ctx, grid, is_gen=False, is_dup=False, fm=None) -> str:
            img_embeds.append(embeds.view(n_ctx, -1))
            gen_flags.append(1 if is_gen else 0)
            dup_flags.append(1 if is_dup else 0)
            grids.append(list(grid))
            if fm is not None:
                fm_list.append(fm)
            return "<img>" + "<IMG_CONTEXT>" * int(n_ctx) + "</img>"

        def _gen_fm(g, dummy=False):
            return {"z": g["z"], "v": g["v"], "t": g["t"], "seq_len": g["seq_len"],
                    "t_eps": g["t_eps"], "dummy": dummy}

        sys_msg = "" if (drop or kind == "s2t") else (self._gen_defaults.get("system_message") or model.system_message)
        user_imgs = list(sample.cond_images or [])
        if drop == "txtimg":
            user_imgs = []
        if kind == "i2t" and not (sample.steps or []):
            user_imgs = []
        user_spans = [
            _reg(*self._und_context_tokens(_pil(im), n_images=len(user_imgs))) for im in user_imgs
        ]
        parts: list[str] = []
        gi = 0
        for idx, st in enumerate(steps):
            if st.text and not drop and body_override is None:
                parts.append(st.text.replace("<image>", ""))
            if st.image is None:
                continue
            pil = _pil(st.image)
            if kind in ("t2i", "i2t"):
                if kind == "t2i" and image_override is not None:
                    if gi >= len(image_override):
                        raise ValueError("image_override shorter than the gen-image chain")
                    pil = _pil(image_override[gi])
                g = self._gen_image_tokens(
                    pil, shared=(shared_noise[gi] if shared_noise is not None else None)
                )
                if capture_noise is not None:
                    capture_noise.append({"u": g["u"], "eps": g["eps"]})
                gi += 1
                parts.append(_reg(g["embeds"], g["seq_len"], g["grid"], is_gen=True, fm=_gen_fm(g)))
                if any((s.text or s.image is not None) for s in steps[idx + 1:]):
                    ue, un, ugrid = self._und_embeds_native(pil)
                    parts.append(_reg(ue, un, ugrid, is_dup=True))
            else:
                ue, un, ugrid = self._und_embeds_native(pil)
                parts.append(_reg(ue, un, ugrid))
        if body_override is not None:
            parts.append(body_override)
        body = "".join(parts)

        q = sample.prompt
        if drop and kind != "i2t":
            q = "<image>" * len(user_spans)
        n_ph = q.count("<image>")
        if n_ph < len(user_spans):
            q = "<image>\n" * (len(user_spans) - n_ph) + q
        for s in user_spans:
            q = q.replace("<image>", s, 1)

        def _compose(answer):
            c = get_conv_template(model.template)
            c.system_message = sys_msg
            c.append_message(c.roles[0], q)
            c.append_message(c.roles[1], answer)
            return c.get_prompt()

        text_blob = body_override if body_override is not None else "".join(s.text or "" for s in steps)
        scaffold = ""
        if not drop and not text_blob.lstrip().startswith("<think>"):
            scaffold = "<think>\n\n</think>\n\n"
        prefix = _compose(None) + scaffold
        full = _compose(scaffold + body)
        sep_id = tok.convert_tokens_to_ids(get_conv_template(model.template).sep.strip())

        ids_t = tok(full, return_tensors="pt").input_ids[0].to(dev)
        ctx_len = int(tok(prefix, return_tensors="pt").input_ids.shape[1])

        dummy_from = None
        if not any(gen_flags):
            g = self._gen_image_tokens(Image.new("RGB", (pm, pm), (127, 127, 127)))
            dummy_span = _reg(g["embeds"], g["seq_len"], g["grid"], is_gen=True, fm=_gen_fm(g, dummy=True))
            dummy_from = int(ids_t.numel())
            ids_t = torch.cat([ids_t, tok(dummy_span, return_tensors="pt").input_ids[0].to(dev)])

        L = ids_t.numel()
        sel = ids_t == IMG_CTX
        assert int(sel.sum()) == sum(int(e.shape[0]) for e in img_embeds), \
            "image span/token count mismatch (text contained literal image tokens?)"

        ind = build_interleaved_indicators(
            ids_t, IMG_START, IMG_CTX, gen_flags, dup_flags, img_end_id=IMG_END
        )
        gen_ind = ind["image_gen_indicators"]

        pos_idx = torch.arange(L, device=dev)
        dup_ind = ind["image_dup_indicators"].to(torch.bool)
        nxt_dup = torch.zeros_like(dup_ind)
        nxt_dup[:-1] = dup_ind[1:]
        dup_img_start = (ids_t == IMG_START) & nxt_dup
        span_dropped = (ids_t == IMG_CTX) | (ids_t == IMG_END) | dup_ind | dup_img_start
        if dummy_from is not None:
            span_dropped = span_dropped | (pos_idx >= dummy_from)
        keep = torch.zeros(L, dtype=torch.bool, device=dev)
        negk = torch.zeros(L, dtype=torch.bool, device=dev)
        if kind in ("t2i", "i2t", "s2t"):
            keep = (pos_idx >= ctx_len) & ~span_dropped
        elif kind == "neg":
            think_id = tok.convert_tokens_to_ids("<think>")
            close_id = tok.convert_tokens_to_ids("</think>")

            def _ws_run(start: int, step: int) -> int:
                n = 0
                i = start
                while 0 <= i < L:
                    t = int(ids_t[i])
                    if t >= int(tok.vocab_size):
                        break
                    if tok.decode([t]).strip() != "":
                        break
                    n += 1
                    i += step
                return n

            opens = ((ids_t == think_id) & (pos_idx >= ctx_len)).nonzero(as_tuple=True)[0]
            if opens.numel():
                i_open = int(opens[0])
                closes = ((ids_t == close_id) & (pos_idx > i_open)).nonzero(as_tuple=True)[0]
                if closes.numel():
                    i_close = int(closes[0])
                    b0 = i_open + 1 + _ws_run(i_open + 1, +1)
                    b1 = i_close - 1 - _ws_run(i_close - 1, -1)
                    t0 = i_close + 1 + _ws_run(i_close + 1, +1)
                    body = (pos_idx >= b0) & (pos_idx <= b1)
                    tail = pos_idx >= t0
                    negk = (body | tail) & (ids_t < int(tok.vocab_size))
            if not bool(negk.any()):
                negk = (pos_idx >= ctx_len) & ~span_dropped & (ids_t < int(tok.vocab_size))

        clm = self._causal_lm()
        emb = clm.get_input_embeddings()(ids_t[None]).clone()
        if img_embeds:
            emb[0][sel] = torch.cat(img_embeds, dim=0).to(emb.dtype)
        grid_hw = torch.tensor(grids, device=dev) if grids else None
        indexes = model.get_thw_indexes(ids_t, grid_hw)
        indexes[0] = ind["token_pos"]
        if self._attn_impl == "flex_attention":
            bm = build_isolation_block_mask(
                ind["document_ids"], ind["modality_indicators"], gen_ind, ind["token_pos"], ind["dup_boundary"]
            )
            attn_mask = {"full_attention": bm, "sliding_attention": bm}
        else:
            attend = build_isolation_attend_mask(
                ind["document_ids"], ind["modality_indicators"], gen_ind, ind["token_pos"], ind["dup_boundary"]
            )
            add = torch.zeros(1, 1, L, L, device=dev, dtype=dt)
            add.masked_fill_(~attend[None, None], torch.finfo(dt).min)
            attn_mask = {"full_attention": add, "sliding_attention": add}

        out = clm.model(
            inputs_embeds=emb, image_gen_indicators=gen_ind[None], indexes=indexes,
            attention_mask=attn_mask, use_cache=False,
        )
        h = out.last_hidden_state
        ctx = {"ids_t": ids_t, "keep": keep, "negk": negk, "fm_list": fm_list,
               "gen_ind": gen_ind, "clm": clm, "kind": kind}
        return h, ctx

    def _fm_term(self, h, ctx, *, global_norm=False):
        torch = self._torch
        F = torch.nn.functional
        fm_list = ctx["fm_list"]
        dist = torch.distributed
        sync = (global_norm and dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1)
        if not fm_list:
            if sync:
                dist.all_reduce(torch.zeros((), device=h.device, dtype=torch.float32),
                                op=dist.ReduceOp.AVG)
            return h.new_zeros(())
        hg = h[0][ctx["gen_ind"]]
        pred_x = self.model.fm_modules["fm_head"](hg)
        zs = torch.cat([f["z"].view(-1, pred_x.shape[-1]) for f in fm_list], dim=0)
        vs = torch.cat([f["v"].view(-1, pred_x.shape[-1]) for f in fm_list], dim=0)
        t_eps = fm_list[0]["t_eps"]
        ts = torch.cat([f["t"].expand(f["seq_len"]) for f in fm_list], dim=0).view(-1, 1)
        pred_v = (pred_x - zs) / (1.0 - ts).clamp_min(t_eps)
        per_tok = F.mse_loss(pred_v.float(), vs.float(), reduction="none").mean(dim=1)
        wts = torch.cat([
            per_tok.new_full((f["seq_len"],), 0.0 if f.get("dummy") else 1.0 / f["seq_len"] ** 0.5)
            for f in fm_list
        ])
        num = (per_tok * wts).sum()
        wsum = wts.sum()
        if sync:
            g = wsum.detach().clone().float()
            dist.all_reduce(g, op=dist.ReduceOp.AVG)
            return num / g.clamp_min(1.0)
        return num / wsum.clamp_min(1e-8)

    def _ce_neg_terms(self, h, ctx):
        torch = self._torch
        F = torch.nn.functional
        ids_t, keep, negk, clm = ctx["ids_t"], ctx["keep"], ctx["negk"], ctx["clm"]
        ce = h.new_zeros(())
        neg = h.new_zeros(())
        if bool(keep.any()) or bool(negk.any()):
            logits = clm.lm_head(h)[0]
            if bool(keep.any()):
                labels = torch.where(keep, ids_t, torch.full_like(ids_t, -100))
                ce = F.cross_entropy(logits[:-1].float(), labels[1:], ignore_index=-100)
            if bool(negk.any()):
                logp = F.log_softmax(logits[:-1].float(), dim=-1)
                tok_logp = logp.gather(1, ids_t[1:].unsqueeze(1)).squeeze(1)
                clamp = float(self._fm.get("unlike_clamp", 0.9))
                p = tok_logp[negk[1:]].exp().clamp(max=clamp)
                neg = -(1.0 - p).log().mean()
        return ce, neg

    def interleaved_teach_loss(self, sample, *, objective="sft"):
        if objective != "sft":
            raise NotImplementedError("only the standard SFT objective is supported")
        from ..training.objective import quadrant_loss_weights

        drop = ""
        if sample.kind == "t2i":
            p_txt = float(self._fm.get("cfg_txt_drop_prob", 0.1))
            p_ti = float(self._fm.get("cfg_txtimg_drop_prob", 0.1))
            r = float(self._torch.rand(()).item())
            if r < p_txt:
                drop = "txt"
            elif r < p_txt + p_ti:
                drop = "txtimg"

        h, ctx = self._teach_forward(sample, drop=drop)
        fm = self._fm_term(h, ctx, global_norm=bool(self._fm.get("fm_global_norm", False)))
        ce, neg = self._ce_neg_terms(h, ctx)
        w = quadrant_loss_weights(ctx["kind"])
        ce_weight = float(self._fm.get("ce_loss_weight", 0.1))
        ce_w = 0.0 if drop else w["ce"] * ce_weight
        loss = w["fm"] * fm + ce_w * ce + w["neg"] * neg
        return {"loss": loss, "fm": fm.detach(), "ce": ce.detach(), "neg": neg.detach(),
                "kind": ctx["kind"], "cfg_drop": drop}

    def trainable_modules(self) -> dict[str, list[str]]:
        return {
            "und": ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"],
            "gen": ["_mot_gen", "fm_modules", "vision_model_mot_gen"],
        }

    def _lora_layers(self) -> list:
        try:
            from peft.tuners.tuners_utils import BaseTunerLayer
        except Exception:
            return []
        return [m for m in self.model.modules() if isinstance(m, BaseTunerLayer)]

    @contextmanager
    def _lora_disabled(self) -> Iterator[None]:
        layers = self._lora_layers()
        if not layers:
            yield
            return
        prev = [getattr(l, "_disable_adapters", False) for l in layers]
        for l in layers:
            l._disable_adapters = True
        try:
            yield
        finally:
            for l, p in zip(layers, prev):
                l._disable_adapters = p

    @contextmanager
    def frozen_base_context(self) -> Iterator[None]:
        layers = self._lora_layers()
        if not layers:
            if not getattr(self, "_warned_no_peft", False):
                self._warned_no_peft = True
                print("[adapter] frozen_base_context: 未找到可关闭的 LoRA, 审查将使用当前权重")
            with self._unwrapped_language_model_context():
                yield
            return
        if not getattr(self, "_logged_peft_ok", False):
            self._logged_peft_ok = True
            print(f"[adapter] frozen_base_context: 审查将关闭 {len(layers)} 个 LoRA 层(冻结底模)")
        with self._lora_disabled():
            with self._unwrapped_language_model_context():
                yield

    @contextmanager
    def unwrapped_context(self) -> Iterator[None]:
        with self._unwrapped_language_model_context():
            yield

    def save_adapter(self, path: str) -> None:
        raise NotImplementedError("save_adapter: implemented in M3")

    def load_adapter(self, path: str) -> None:
        raise NotImplementedError("load_adapter: implemented in M3")

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def gen_defaults(self) -> dict:
        return dict(self._gen_defaults)
