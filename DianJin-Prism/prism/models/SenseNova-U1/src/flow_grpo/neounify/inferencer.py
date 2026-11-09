"""
NeoUnifyInferencer: Wraps NEOChatModel for Flow-GRPO training.

Provides the same interface as the Bagel InterleaveInferencer:
  - __call__(text, learn=False, ...) -> dict
    - Sampling mode: returns {image, all_latents, all_log_probs, timesteps}
    - Training mode: returns {clipfrac, clipfrac_gt_one, clipfrac_lt_one, policy_loss, kl_loss, loss}
"""
import math
import random
from typing import Optional, Tuple, Dict, Any

import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
from functools import partial


def randn_tensor(shape, generator=None, device=None, dtype=None):
    """Generate a random tensor (replaces diffusers.utils.torch_utils.randn_tensor)."""
    if generator is not None:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)

tqdm = partial(tqdm, dynamic_ncols=True, leave=False, position=1)


class NeoUnifyInferencer:
    """Wraps a NEOChatModel for Flow-GRPO sampling and training."""

    def __init__(self, model, tokenizer, compute_dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            model: NEOChatModel instance.
            tokenizer: matching tokenizer.
            compute_dtype: 主计算精度（默认 bfloat16，与官方 it2i_generate / SFT 一致）。
                ODE step / prefix forward / v_pred forward / 初始 noise 均使用该 dtype。
                SDE step 内部为避免 bf16 overflow (参考 sd3_sde_with_logprob.py),
                强制走 fp32 计算 prev_sample_mean / log_prob, 完成后转回 compute_dtype.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.compute_dtype = compute_dtype
        # Cached model attributes
        self.patch_size = model.patch_size
        self.downsample_ratio = model.config.downsample_ratio
        self.merge_size = int(1 / self.downsample_ratio)

    def _get_inner_lm_model(self):
        """Get the actual inner transformer model (Qwen3Model/Qwen3MoeModel).
        
        NOTE: Only use for NON-FSDP mode (single GPU without sharding).
        In FSDP mode, use _prefix_forward/_predict_v_forward which go through
        the FSDP-wrapped model.language_model instead.
        """
        lm = self.model.language_model
        if hasattr(lm, 'base_model') and hasattr(lm.base_model, 'model'):
            inner_causal_lm = lm.base_model.model
            if hasattr(inner_causal_lm, 'model'):
                return inner_causal_lm.model
            return inner_causal_lm
        else:
            return lm.model

    def _prefix_forward(self, input_ids, indexes, attention_mask):
        """FSDP/PEFT-compatible prefix forward.
        
        Calls through model.language_model (FSDP-wrapped) to ensure parameters
        are properly all-gathered before use. Qwen3ForCausalLM.forward returns
        CausalLMOutputWithPast where .hidden_states = last_hidden_state.
        """
        lm = self.model.language_model
        out = lm(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
        )
        # Qwen3ForCausalLM stores outputs.last_hidden_state directly in .hidden_states
        return out.past_key_values, out.hidden_states

    def _predict_v_forward(self, input_embeds, indexes_image, attn_mask, past_key_values, 
                           t, z, image_token_num, timestep_embeddings=None, image_size=None,
                           use_cache=True, **kwargs):
        """FSDP/PEFT-compatible v prediction forward via KV cache."""
        model = self.model
        lm = model.language_model
        B, L = z.shape[0], z.shape[1]

        out = lm(
            inputs_embeds=input_embeds,
            image_gen_indicators=torch.ones(
                (input_embeds.shape[0], input_embeds.shape[1]),
                dtype=torch.bool, device=input_embeds.device
            ),
            indexes=indexes_image,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=use_cache,
        )
        hidden_states = out.hidden_states

        # FSDP mixed precision may output float32, cast back to model dtype
        model_dtype = next(model.fm_modules.parameters()).dtype
        if hidden_states.dtype != model_dtype:
            hidden_states = hidden_states.to(model_dtype)

        if model.use_pixel_head:
            merge_size = int(1 / model.downsample_ratio)
            token_h = image_size[1] // (model.patch_size * merge_size)
            token_w = image_size[0] // (model.patch_size * merge_size)

            img_reshaped = hidden_states[:, -image_token_num:].view(B, token_h, token_w, -1)
            img_2d = torch.einsum("b h w c -> b c h w", img_reshaped)
            img_2d = img_2d.contiguous().view(B, -1, token_h, token_w)
            smoothed_img_2d = model.fm_modules['fm_head'](img_2d)
            smoothed_reshaped = smoothed_img_2d.view(
                B, 3, token_h, model.patch_size * merge_size, token_w, model.patch_size * merge_size
            )
            smoothed_reshaped = torch.einsum("b c h p w q -> b h w p q c", smoothed_reshaped)
            x_pred = smoothed_reshaped.contiguous().view(
                B, L, model.patch_size * merge_size * model.patch_size * merge_size * 3
            )
        else:
            if model.use_deep_fm_head:
                x_pred = model.fm_modules["fm_head"](
                    hidden_states[:, -image_token_num:].view(B*L, -1),
                    t.repeat(B*L)
                ).view(B, L, -1)
            else:
                x_pred = model.fm_modules["fm_head"](
                    hidden_states[:, -image_token_num:].view(B, L, -1)
                ).view(B, L, -1)

        v_pred = (x_pred - z) / (1 - t).clamp_min(model.config.t_eps)
        return v_pred

    # ------------------------------------------------------------------
    # Public interface (matches Bagel InterleaveInferencer.__call__)
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: str,
        learn: bool = False,
        sample: Optional[Dict] = None,
        grpo_config=None,
        accelerator=None,
        optimizer=None,
        transformer=None,
        num_timesteps: int = 10,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: Tuple[float, float] = (0, 1),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        timestep_shift: float = 3.0,
        image_shapes: Tuple[int, int] = (1024, 1024),
        noise_level: float = 0.7,
        generators=None,
        images=None,  # IT2I mode: list of PIL.Image; None/[] = T2I mode
        **kwargs,
    ) -> Dict[str, Any]:
        if learn:
            return self._generate_image_learn(
                text=text,
                sample=sample,
                grpo_config=grpo_config,
                accelerator=accelerator,
                optimizer=optimizer,
                transformer=transformer,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                image_shapes=image_shapes,
                noise_level=noise_level,
                images=images,
            )
        else:
            return self._generate_image_sample(
                text=text,
                grpo_config=grpo_config,
                accelerator=accelerator,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                image_shapes=image_shapes,
                noise_level=noise_level,
                generators=generators,
                images=images,
            )

    # ------------------------------------------------------------------
    # Prefix / KV cache construction
    # ------------------------------------------------------------------

    def _build_prefix_caches(self, text: str, image_shapes: Tuple[int, int], training: bool = False,
                              images=None, cfg_text_scale: float = 1.0, cfg_img_scale: float = 1.0):
        """Build prefix KV caches for CFG (supports both t2i and it2i modes).

        Args:
            text: prompt text (may contain <image> placeholders for it2i).
            image_shapes: target output (W, H).
            training: skip flash KV buffer prep when True (inplace issue with autograd).
            images: list of PIL images for it2i mode. None/[] = t2i mode.
            cfg_text_scale, cfg_img_scale: decide which CFG branches to compute.

        Returns dict with:
          past_kv_cond, past_kv_img_cond, past_kv_uncond,
          indexes_image_cond, indexes_image_img_cond, indexes_image_uncond,
          token_h, token_w, is_it2i
        Some entries may be None when their CFG branch isn't needed.
        For backward compatibility we also keep `past_kv_tu` / `indexes_image_tu`.
        """
        model = self.model
        tokenizer = self.tokenizer
        device = next(model.parameters()).device

        from sensenova_u1.models.neo_unify.modeling_neo_chat import prepare_flash_kv_cache
        from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN, load_image_native

        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

        is_it2i = images is not None and len(images) > 0
        needs_cfg = not (cfg_text_scale == 1 and cfg_img_scale == 1)
        needs_img_condition = needs_cfg and (cfg_img_scale == 1 or cfg_text_scale != cfg_img_scale)
        needs_uncondition = needs_cfg and cfg_img_scale != 1

        token_h = image_shapes[1] // (self.patch_size * self.merge_size)
        token_w = image_shapes[0] // (self.patch_size * self.merge_size)

        # ============================================================
        # IT2I path
        # ============================================================
        if is_it2i:
            # Set img_context_token_id (read by _build_it2i_inputs to locate image tokens)
            model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

            # Inject <image> placeholders if missing
            prompt = text
            image_token_count = prompt.count('<image>')
            if len(images) > image_token_count:
                if image_token_count == 0 and len(images) > 1:
                    prompt = "".join(f"Image-{i+1}:<image>\n" for i in range(len(images))) + prompt
                else:
                    prompt = "<image>\n" * (len(images) - image_token_count) + prompt

            # 与官方 inference.py _resize_to_max_budget 完全一致:
            # 使用 min_pixels=max_pixels=input_max_pixels, 小图也会上采样
            from sensenova_u1.models.neo_unify.utils import smart_resize
            DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
            input_max_pixels = DEFAULT_INPUT_MAX_PIXELS
            resized_images = []
            for img in images:
                w, h = img.size
                factor = model.patch_size  # _IMAGE_GRID_FACTOR = patch_size
                resized_h, resized_w = smart_resize(
                    height=h, width=w, factor=factor,
                    min_pixels=input_max_pixels, max_pixels=input_max_pixels,
                )
                if (resized_w, resized_h) != (w, h):
                    img = img.resize((resized_w, resized_h), Image.LANCZOS)
                resized_images.append(img)
            images = resized_images

            # Load images via official preprocessing (load_image_native)
            pixel_values, grid_hw = [], []
            for image in images:
                cur_pv, cur_ghw = load_image_native(
                    image, model.patch_size, model.downsample_ratio,
                    min_pixels=512*512,
                    max_pixels=min(2048*2048, (4096*4096) // len(images)),
                    upscale=False,
                )
                pixel_values.append(cur_pv.to(device).to(torch.bfloat16))
                grid_hw.append(cur_ghw.to(device))
            pixel_values = torch.cat(pixel_values)
            grid_hw_tensor = torch.cat(grid_hw)

            # Build queries (note: think is closed)
            think_content = '<think>\n\n</think>\n\n' + IMG_START_TOKEN
            query_cond = model._build_t2i_query(
                prompt, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content
            )
            query_img_cond = (
                model._build_t2i_query('<image>' * len(images), append_text=IMG_START_TOKEN)
                if needs_img_condition else None
            )
            query_uncond = (
                model._build_t2i_query("", append_text=IMG_START_TOKEN)
                if needs_uncondition else None
            )

            # Replace each <image> placeholder with full image-token string
            for i in range(grid_hw_tensor.shape[0]):
                num_patch = int(grid_hw_tensor[i, 0] * grid_hw_tensor[i, 1] * model.downsample_ratio**2)
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch + IMG_END_TOKEN
                query_cond = query_cond.replace('<image>', image_tokens, 1)
                if query_img_cond is not None:
                    query_img_cond = query_img_cond.replace('<image>', image_tokens, 1)

            # Build it2i inputs (text + image embeddings)
            # DeepSpeed wraps language_model → temporarily unwrap for get_input_embeddings()
            _orig_lm = model.language_model
            if hasattr(_orig_lm, 'module'):
                model.language_model = _orig_lm.module
            try:
                input_embeds_cond, indexes_cond, attn_mask_cond = model._build_it2i_inputs(
                    tokenizer, query_cond, pixel_values, grid_hw_tensor
                )
                input_embeds_img_cond = indexes_img_cond = attn_mask_img_cond = None
                if query_img_cond is not None:
                    input_embeds_img_cond, indexes_img_cond, attn_mask_img_cond = model._build_it2i_inputs(
                        tokenizer, query_img_cond, pixel_values, grid_hw_tensor
                    )
                input_embeds_uncond = indexes_uncond = attn_mask_uncond = None
                if query_uncond is not None:
                    # Uncondition: text-only, no image embeds
                    input_embeds_uncond, indexes_uncond, attn_mask_uncond = model._build_it2i_inputs(
                        tokenizer, query_uncond
                    )
            finally:
                model.language_model = _orig_lm

            # Image indexes for denoising step
            indexes_image_cond = model._build_t2i_image_indexes(
                token_h, token_w, indexes_cond[0].max() + 1, device=device
            )
            indexes_image_img_cond = (
                model._build_t2i_image_indexes(
                    token_h, token_w, indexes_img_cond[0].max() + 1, device=device
                ) if indexes_img_cond is not None else None
            )
            indexes_image_uncond = (
                model._build_t2i_image_indexes(
                    token_h, token_w, indexes_uncond[0].max() + 1, device=device
                ) if indexes_uncond is not None else None
            )

            # Prefix forward via inputs_embeds path
            past_kv_cond, _ = self._prefix_forward_it2i(input_embeds_cond, indexes_cond, attn_mask_cond)
            past_kv_img_cond = None
            if input_embeds_img_cond is not None:
                past_kv_img_cond, _ = self._prefix_forward_it2i(
                    input_embeds_img_cond, indexes_img_cond, attn_mask_img_cond
                )
            past_kv_uncond = None
            if input_embeds_uncond is not None:
                past_kv_uncond, _ = self._prefix_forward_it2i(
                    input_embeds_uncond, indexes_uncond, attn_mask_uncond
                )

            if not training:
                prepare_flash_kv_cache(past_kv_cond, current_len=token_h * token_w, batch_size=1)
                if past_kv_img_cond is not None:
                    prepare_flash_kv_cache(past_kv_img_cond, current_len=token_h * token_w, batch_size=1)
                if past_kv_uncond is not None:
                    prepare_flash_kv_cache(past_kv_uncond, current_len=token_h * token_w, batch_size=1)

            return {
                "past_kv_cond": past_kv_cond,
                "past_kv_img_cond": past_kv_img_cond,
                "past_kv_uncond": past_kv_uncond,
                "indexes_image_cond": indexes_image_cond,
                "indexes_image_img_cond": indexes_image_img_cond,
                "indexes_image_uncond": indexes_image_uncond,
                # Backward compat with t2i CFG branch (text-uncond == img_cond when img_cfg=1)
                "past_kv_tu": past_kv_img_cond if past_kv_img_cond is not None else past_kv_uncond,
                "indexes_image_tu": indexes_image_img_cond if indexes_image_img_cond is not None else indexes_image_uncond,
                "token_h": token_h,
                "token_w": token_w,
                "is_it2i": True,
            }

        # ============================================================
        # T2I path (original)
        # ============================================================
        query_cond_t2i = model._build_t2i_query(
            text, system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text='<think>\n\n</think>\n\n' + IMG_START_TOKEN
        )
        input_ids_cond, indexes_cond, attn_mask_cond = model._build_t2i_text_inputs(tokenizer, query_cond_t2i)

        query_tu = model._build_t2i_query("", append_text=IMG_START_TOKEN)
        input_ids_tu, indexes_tu, attn_mask_tu = model._build_t2i_text_inputs(tokenizer, query_tu)

        indexes_image_cond = model._build_t2i_image_indexes(
            token_h, token_w, indexes_cond[0].max() + 1, device=device
        )
        indexes_image_tu = model._build_t2i_image_indexes(
            token_h, token_w, indexes_tu[0].max() + 1, device=device
        )

        past_kv_cond, _ = self._prefix_forward(input_ids_cond, indexes_cond, attn_mask_cond)
        past_kv_tu, _ = self._prefix_forward(input_ids_tu, indexes_tu, attn_mask_tu)

        if not training:
            prepare_flash_kv_cache(past_kv_cond, current_len=token_h * token_w, batch_size=1)
            prepare_flash_kv_cache(past_kv_tu, current_len=token_h * token_w, batch_size=1)

        return {
            "past_kv_cond": past_kv_cond,
            "past_kv_tu": past_kv_tu,
            "past_kv_img_cond": None,
            "past_kv_uncond": past_kv_tu,
            "indexes_image_cond": indexes_image_cond,
            "indexes_image_tu": indexes_image_tu,
            "indexes_image_img_cond": None,
            "indexes_image_uncond": indexes_image_tu,
            "token_h": token_h,
            "token_w": token_w,
            "is_it2i": False,
        }

    def _prefix_forward_it2i(self, input_embeds, indexes, attention_mask):
        """FSDP/PEFT-compatible it2i prefix forward (inputs_embeds path)."""
        lm = self.model.language_model
        out = lm(
            inputs_embeds=input_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return out.past_key_values, out.hidden_states

    # ------------------------------------------------------------------
    # Single flow step: v_pred computation
    # ------------------------------------------------------------------

    def _forward_flow(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        prefix_caches: dict,
        image_shapes: Tuple[int, int],
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: Tuple[float, float] = (0, 1),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Compute velocity prediction v_pred for a given latent z at time t.
        
        Args:
            use_cache: Pass False during training (gradient checkpointing compat).
        """
        model = self.model
        device = z.device

        token_h = prefix_caches["token_h"]
        token_w = prefix_caches["token_w"]
        grid_h = image_shapes[1] // self.patch_size
        grid_w = image_shapes[0] // self.patch_size
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device)

        # Compute noise_scale for embeddings
        noise_scale = self._compute_noise_scale(grid_h, grid_w)

        # Prepare image embeddings from z
        # z 可能是 fp32 (SDE 后) 或 bf16 (ODE)，cast 到 model.dtype 进入模型
        model_dtype = next(model.fm_modules.parameters()).dtype
        image_prediction = model.unpatchify(z, self.patch_size * self.merge_size, image_shapes[1], image_shapes[0])
        image_input = model.patchify(image_prediction, self.patch_size, channel_first=True)
        image_embeds = model.extract_feature(
            image_input.view(1 * grid_h * grid_w, -1).to(model_dtype), gen_model=True, grid_hw=gen_grid_hw
        ).view(1, token_h * token_w, -1)

        # Timestep embeddings
        t_expanded = t.expand(token_h * token_w)
        timestep_embeddings = model.fm_modules['timestep_embedder'](t_expanded).view(1, token_h * token_w, -1)
        if model.add_noise_scale_embedding:
            noise_scale_tensor = torch.full_like(t_expanded, noise_scale / model.noise_scale_max_value)
            noise_embeddings = model.fm_modules['noise_scale_embedder'](noise_scale_tensor).view(1, token_h * token_w, -1)
            timestep_embeddings = timestep_embeddings + noise_embeddings
        image_embeds = image_embeds + timestep_embeddings

        # Condition prediction
        attention_mask_none = {"full_attention": None}
        out_cond = self._predict_v_forward(
            image_embeds, prefix_caches["indexes_image_cond"],
            attention_mask_none, prefix_caches["past_kv_cond"],
            t, z, image_token_num=token_h * token_w,
            timestep_embeddings=timestep_embeddings,
            image_size=image_shapes,
            use_cache=use_cache,
        )

        # Apply CFG (supports t2i 2-branch and it2i 3-branch CFG)
        use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
        if not use_cfg or (cfg_text_scale == 1 and cfg_img_scale == 1):
            v_pred = out_cond
        else:
            is_it2i = prefix_caches.get("is_it2i", False)
            past_kv_img_cond = prefix_caches.get("past_kv_img_cond")
            past_kv_uncond = prefix_caches.get("past_kv_uncond")
            indexes_image_img_cond = prefix_caches.get("indexes_image_img_cond")
            indexes_image_uncond = prefix_caches.get("indexes_image_uncond")

            # Pick CFG formula matching official it2i_generate:
            #   img_cfg=1     : v = out_img_cond + cfg_text * (out_cond - out_img_cond)
            #   text=img_cfg  : v = out_uncond + cfg_text * (out_cond - out_uncond)
            #   else          : v = out_uncond + cfg_text*(out_cond-out_img_cond) + cfg_img*(out_img_cond-out_uncond)
            if is_it2i and cfg_img_scale == 1 and past_kv_img_cond is not None:
                out_img_cond = self._predict_v_forward(
                    image_embeds, indexes_image_img_cond,
                    attention_mask_none, past_kv_img_cond,
                    t, z, image_token_num=token_h * token_w,
                    timestep_embeddings=timestep_embeddings,
                    image_size=image_shapes,
                    use_cache=use_cache,
                )
                v_pred = out_img_cond + cfg_text_scale * (out_cond - out_img_cond)
                ref_for_renorm = out_cond
            elif is_it2i and cfg_text_scale != cfg_img_scale and past_kv_img_cond is not None and past_kv_uncond is not None:
                out_img_cond = self._predict_v_forward(
                    image_embeds, indexes_image_img_cond,
                    attention_mask_none, past_kv_img_cond,
                    t, z, image_token_num=token_h * token_w,
                    timestep_embeddings=timestep_embeddings,
                    image_size=image_shapes,
                    use_cache=use_cache,
                )
                out_uncond = self._predict_v_forward(
                    image_embeds, indexes_image_uncond,
                    attention_mask_none, past_kv_uncond,
                    t, z, image_token_num=token_h * token_w,
                    timestep_embeddings=timestep_embeddings,
                    image_size=image_shapes,
                    use_cache=use_cache,
                )
                v_pred = (out_uncond
                          + cfg_text_scale * (out_cond - out_img_cond)
                          + cfg_img_scale * (out_img_cond - out_uncond))
                ref_for_renorm = out_cond
            else:
                # T2I 2-branch CFG (original path)
                out_uncond = self._predict_v_forward(
                    image_embeds, prefix_caches["indexes_image_tu"],
                    attention_mask_none, prefix_caches["past_kv_tu"],
                    t, z, image_token_num=token_h * token_w,
                    timestep_embeddings=timestep_embeddings,
                    image_size=image_shapes,
                    use_cache=use_cache,
                )
                v_pred = out_uncond + cfg_text_scale * (out_cond - out_uncond)
                ref_for_renorm = out_cond

            # CFG renorm
            if cfg_text_scale > 1 and use_cfg:
                if cfg_renorm_type == 'global':
                    norm_v_cond = torch.norm(ref_for_renorm, dim=(1, 2), keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                    scale = (norm_v_cond / (norm_v_cfg + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_pred = v_pred * scale
                elif cfg_renorm_type == 'channel':
                    norm_v_cond = torch.norm(ref_for_renorm, dim=-1, keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (norm_v_cond / (norm_v_cfg + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_pred = v_pred * scale

        return v_pred

    # ------------------------------------------------------------------
    # SDE step with log-prob (adapted for NEO-Unify with noise_scale correction)
    # ------------------------------------------------------------------

    def _sde_step_with_logprob(
        self,
        v_pred: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        z: torch.Tensor,
        noise_level: float = 0.7,
        noise_scale: float = 1.0,
        prev_sample: torch.Tensor = None,
        generator=None,
    ):
        """
        SDE step with log-probability for SenseNova-U1 flow matching.

        Adapts flow_grpo's SDE to U1's conventions with noise_scale correction.

        Key insight: U1 initializes z_0 = noise_scale * N(0,I) where noise_scale
        can be > 1 (e.g. 2.0 at 2048x2048). The marginal is
            p_t(z) = N(t*x1, sigma^2 * noise_scale^2 * I)
        so the score has an extra 1/noise_scale^2 factor vs the standard formula.
        Without this correction the drift term is too large and kills the signal.

        NEO convention: t goes 0(noise)->1(clean), v_pred points towards clean.
        When noise_level=0 (g=0): z_mean = z + v_pred*dt (matches Euler ODE).

        Args:
            v_pred: velocity prediction, shape (B, L, D)
            t: current timestep scalar (0→1)
            t_next: next timestep scalar (t_next > t)
            z: current patchified latent, same shape as v_pred
            noise_level: controls SDE exploration magnitude
            noise_scale: model's resolution-dependent noise_scale
            prev_sample: if provided, use this as z_next instead of sampling
            generator: optional torch Generator for reproducibility

        Returns:
            z_next: next latent sample
            log_prob: log probability, scalar (mean over all dims)
            z_mean: deterministic mean of the transition
            std_dev_t: diffusion coefficient
        """
        # 主计算精度取决于调用者 (默认 self.compute_dtype = bf16)
        # SDE 内部强制走 fp32 (参考 sd3_sde_with_logprob.py:36 注释:
        # "bf16 can overflow here when compute prev_sample_mean")
        # 输出时转回 orig_dtype保证与外部全程 dtype 一致
        orig_dtype = z.dtype
        device = z.device

        # ODE 快速路径 (noise_level=0)：与官方 it2i_generate line 1655 完全一致
        # 不走 fp32, 保持 orig_dtype (默认 bf16)
        if noise_level == 0 and prev_sample is None:
            dt = t_next - t
            z_next = z + dt * v_pred
            log_prob = torch.zeros(z.shape[0], device=device, dtype=orig_dtype)
            std_dev_t = torch.zeros((), device=device, dtype=orig_dtype)
            return z_next, log_prob, z_next, std_dev_t

        # SDE 路径：参考 sd3_sde_with_logprob.py，全程 fp32 避免 overflow
        v_pred_f = v_pred.float()
        z_f = z.float()
        prev_sample_f = prev_sample.float() if prev_sample is not None else None

        dt = t_next - t  # positive (t increasing: 0→1)
        sigma = (1.0 - t).clamp(min=1e-5)  # noise fraction at current t

        # Diffusion coefficient
        std_dev_t = torch.sqrt(sigma / (1.0 - sigma)).clamp(max=5.0) * noise_level

        # SDE drift-corrected mean (全程 fp32)
        # Score correction divided by noise_scale² to account for non-unit initial noise
        g2 = std_dev_t ** 2
        ns2 = float(noise_scale) ** 2
        z_mean = z_f * (1.0 - g2 / (2.0 * sigma * ns2) * dt) + \
                 v_pred_f * (1.0 + g2 * (1.0 - sigma) / (2.0 * sigma * ns2)) * dt

        # Noise standard deviation for this step
        step_noise_scale = std_dev_t * torch.sqrt(dt)

        if prev_sample_f is None:
            noise = randn_tensor(
                z_f.shape, generator=generator, device=device, dtype=torch.float32
            )
            z_next_f = z_mean + step_noise_scale * noise
        else:
            z_next_f = prev_sample_f

        # Log probability under the Gaussian transition kernel (fp32)
        log_prob = (
            -((z_next_f.detach() - z_mean) ** 2) / (2.0 * step_noise_scale ** 2)
            - torch.log(step_noise_scale)
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        # 对齐 bagel 策略：SDE 输出保持 fp32 不转回
        # - z_next fp32: 后续 _forward_flow 内 extract_feature 自动 cast 到 model.dtype
        # - log_prob fp32: GRPO ratio 需要精度
        # - all_latents 存 fp32: 训练 replay 精确无损
        return z_next_f, log_prob, z_mean, std_dev_t

    # ------------------------------------------------------------------
    # Sampling mode: generate image and record trajectory
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_image_sample(
        self,
        text: str,
        grpo_config,
        accelerator,
        num_timesteps: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Tuple[float, float],
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        timestep_shift: float,
        image_shapes: Tuple[int, int],
        noise_level: float,
        generators=None,
        images=None,
    ) -> Dict[str, Any]:
        model = self.model
        device = next(model.parameters()).device

        # Build prefix caches (auto-detects t2i / it2i based on `images`)
        prefix_caches = self._build_prefix_caches(
            text, image_shapes, training=False,
            images=images, cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
        )
        token_h = prefix_caches["token_h"]
        token_w = prefix_caches["token_w"]

        # Compute noise_scale (resolution-dependent)
        grid_h = image_shapes[1] // self.patch_size
        grid_w = image_shapes[0] // self.patch_size
        noise_scale = self._compute_noise_scale(grid_h, grid_w)

        if generators:
            # 确保 generator 在正确设备上
            gen = generators[0]
            if gen.device.type != device.type if hasattr(device, 'type') else gen.device.type != 'cuda':
                gen = torch.Generator(device=device).manual_seed(gen.initial_seed())
            initial_noise = noise_scale * torch.randn(
                (1, 3, image_shapes[1], image_shapes[0]),
                device=device, dtype=self.compute_dtype, generator=gen
            )
        else:
            initial_noise = noise_scale * torch.randn(
                (1, 3, image_shapes[1], image_shapes[0]),
                device=device, dtype=self.compute_dtype
            )

        # Patchify initial noise to get z
        z = model.patchify(initial_noise, self.patch_size * self.merge_size)

        # Setup timesteps (NEO convention: 0->1)
        timesteps = torch.linspace(0.0, 1.0, num_timesteps + 1, device=device)
        timesteps = model._apply_time_schedule(timesteps, token_h * token_w, timestep_shift)

        # SDE window setup
        sde_window_size = grpo_config.sample.sde_window_size
        sde_window_range = grpo_config.sample.sde_window_range
        process_index = getattr(accelerator, 'process_index', 0)
        random.seed(process_index)
        sde_begin = random.randint(sde_window_range[0], sde_window_range[1] - sde_window_size)

        all_latents = []
        all_log_probs = []
        all_timesteps = []

        for i in tqdm(range(num_timesteps), total=num_timesteps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Determine noise level for this step
            if i < sde_begin:
                cur_noise_level = 0
            elif i == sde_begin:
                cur_noise_level = noise_level
                all_latents.append(z.clone())
            elif i > sde_begin and i < sde_begin + sde_window_size:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0

            # Forward flow to get v_pred
            v_pred = self._forward_flow(
                z, t, prefix_caches, image_shapes,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            )

            # SDE step with noise_scale correction
            z_next, log_prob, _, _ = self._sde_step_with_logprob(
                v_pred, t, t_next, z,
                noise_level=cur_noise_level,
                noise_scale=noise_scale,
            )
            # SDE step 已保证返回原 dtype，无需 .to() 转换
            z = z_next

            # Record trajectory in SDE window
            if i >= sde_begin and i < sde_begin + sde_window_size:
                all_latents.append(z.clone())
                all_log_probs.append(log_prob)
                all_timesteps.append(t.item())

        # Clear flash KV caches (handles both t2i and it2i)
        from sensenova_u1.models.neo_unify.modeling_neo_chat import clear_flash_kv_cache
        for k in ("past_kv_cond", "past_kv_img_cond", "past_kv_uncond", "past_kv_tu"):
            cache = prefix_caches.get(k)
            if cache is not None:
                clear_flash_kv_cache(cache)

        # Decode final image
        image = self._decode_image(z, image_shapes)

        all_timesteps_tensor = torch.tensor(all_timesteps, device=device)
        return {
            "image": image,
            "all_latents": all_latents,
            "all_log_probs": all_log_probs,
            "timesteps": all_timesteps_tensor,
        }

    # ------------------------------------------------------------------
    # Training mode: replay trajectory and compute GRPO loss
    # ------------------------------------------------------------------

    def _generate_image_learn(
        self,
        text: str,
        sample: Dict,
        grpo_config,
        accelerator,
        optimizer,
        transformer,
        num_timesteps: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Tuple[float, float],
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        timestep_shift: float,
        image_shapes: Tuple[int, int],
        noise_level: float,
        images=None,
    ) -> Dict[str, Any]:
        model = self.model
        device = next(model.parameters()).device

        # Build prefix caches (no grad for prefix, training=True to skip flash buffer)
        with torch.no_grad():
            prefix_caches = self._build_prefix_caches(
                text, image_shapes, training=True,
                images=images, cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
            )
        token_h = prefix_caches["token_h"]
        token_w = prefix_caches["token_w"]

        # Compute noise_scale (resolution-dependent)
        grid_h = image_shapes[1] // self.patch_size
        grid_w = image_shapes[0] // self.patch_size
        noise_scale = self._compute_noise_scale(grid_h, grid_w)

        latents = sample["latents"]
        prev_latents = sample["prev_latents"]
        timesteps_recorded = sample["timesteps"]

        # Reconstruct full timestep schedule for dt computation
        full_timesteps = torch.linspace(0.0, 1.0, num_timesteps + 1, device=device)
        full_timesteps = model._apply_time_schedule(full_timesteps, token_h * token_w, timestep_shift)

        advantages = torch.clamp(
            sample["advantages"],
            -grpo_config.train.adv_clip_max,
            grpo_config.train.adv_clip_max,
        )

        clipfrac_list = []
        clipfrac_gt_one_list = []
        clipfrac_lt_one_list = []
        policy_loss_list = []
        policy_loss_abs_list = []
        kl_loss_list = []
        loss_list = []
        loss_abs_list = []
        ratio_mean_list = []
        ratio_std_list = []
        ratio_abs_deviation_mean_list = []
        old_logprob_mean_list = []
        new_logprob_mean_list = []
        logprob_delta_mean_list = []
        logprob_delta_abs_mean_list = []
        advantage_mean_list = []
        advantage_abs_mean_list = []
        loss_scale_list = []
        sync_grad_list = []

        # Check if LoRA is enabled (for KL via disable_adapter)
        # NOTE: accelerator.prepare() wraps model in DeepSpeedEngine/FSDP,
        # so we need to unwrap to find the real PEFT model.
        _inner_lm = model.language_model
        if hasattr(_inner_lm, 'module'):  # DeepSpeedEngine / FSDP wrapper
            _inner_lm = _inner_lm.module
        use_lora = hasattr(_inner_lm, 'disable_adapter_layers')

        for i, t in tqdm(
            enumerate(timesteps_recorded),
            total=len(timesteps_recorded),
            disable=not accelerator.is_local_main_process,
        ):
            t_tensor = torch.tensor(t, device=device) if not isinstance(t, torch.Tensor) else t.to(device)

            # Find corresponding t_next from full schedule
            t_index = (full_timesteps[:-1] - t_tensor).abs().argmin()
            t_next = full_timesteps[t_index + 1]

            with accelerator.accumulate(transformer):
                # Forward flow with gradients (current policy)
                # use_cache=False for gradient checkpointing compatibility
                v_pred = self._forward_flow(
                    latents[i].to(device), t_tensor, prefix_caches, image_shapes,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    use_cache=False,
                )

                # Compute log_prob for the recorded next state
                _, log_prob, z_mean, std_dev_t = self._sde_step_with_logprob(
                    v_pred, t_tensor, t_next, latents[i].to(device),
                    noise_level=noise_level,
                    noise_scale=noise_scale,
                    prev_sample=prev_latents[i].to(device),
                )

                # KL loss via disable_adapter (reference = base model without LoRA)
                if grpo_config.train.beta > 0 and use_lora:
                    with torch.no_grad():
                        _inner_lm.disable_adapter_layers()
                        v_pred_ref = self._forward_flow(
                            latents[i].to(device), t_tensor, prefix_caches, image_shapes,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            use_cache=False,
                        )
                        _, _, z_mean_ref, _ = self._sde_step_with_logprob(
                            v_pred_ref, t_tensor, t_next, latents[i].to(device),
                            noise_level=noise_level,
                            noise_scale=noise_scale,
                            prev_sample=prev_latents[i].to(device),
                        )
                        _inner_lm.enable_adapter_layers()
                    kl_loss = ((z_mean - z_mean_ref) ** 2).mean() / (2 * std_dev_t ** 2)
                else:
                    kl_loss = torch.tensor(0.0, device=device)

                # GRPO loss computation
                old_log_prob = sample["log_probs"][i].to(device)
                logprob_delta = log_prob - old_log_prob
                ratio = torch.exp(logprob_delta)
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - grpo_config.train.clip_range_lt,
                    1.0 + grpo_config.train.clip_range_gt,
                )
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                if getattr(grpo_config, 'scale_loss_by_resolution', False):
                    N = sample["latents"][i].numel()  # B × L × D, B=1 so this gives L × D
                    N_ref = getattr(grpo_config, 'log_prob_n_ref', 65536)
                    loss_scale = (float(N) / float(N_ref)) ** 0.5
                else:
                    loss_scale = 1.0

                loss = policy_loss * loss_scale + grpo_config.train.beta * kl_loss

                # Backward + optimizer step (DeepSpeed 内部管理梯度)
                accelerator.backward(loss)
                optimizer.step()
                # 从 DeepSpeed engine 获取 grad_norm（内部 allreduce 后的值）
                _raw_gn = getattr(transformer, '_global_grad_norm', 0.0) or 0.0
                if isinstance(_raw_gn, torch.Tensor):
                    _cur_grad_norm = _raw_gn.detach().clone().to(device)
                else:
                    _cur_grad_norm = torch.tensor(float(_raw_gn), device=device)
                optimizer.zero_grad()

            # Record metrics
            clipfrac_list.append(
                torch.mean(((ratio - 1.0).abs() > grpo_config.train.clip_range_gt).float())
            )
            clipfrac_gt_one_list.append(
                torch.mean((ratio - 1.0 > grpo_config.train.clip_range_gt).float())
            )
            clipfrac_lt_one_list.append(
                torch.mean((1.0 - ratio > grpo_config.train.clip_range_lt).float())
            )
            policy_loss_list.append(policy_loss.detach())
            policy_loss_abs_list.append(policy_loss.detach().abs())
            kl_loss_list.append(kl_loss.detach())
            loss_list.append(loss.detach())
            loss_abs_list.append(loss.detach().abs())
            ratio_mean_list.append(ratio.detach().mean())
            ratio_std_list.append(ratio.detach().std(unbiased=False))
            ratio_abs_deviation_mean_list.append((ratio.detach() - 1.0).abs().mean())
            old_logprob_mean_list.append(old_log_prob.detach().mean())
            new_logprob_mean_list.append(log_prob.detach().mean())
            logprob_delta_mean_list.append(logprob_delta.detach().mean())
            logprob_delta_abs_mean_list.append(logprob_delta.detach().abs().mean())
            advantage_mean_list.append(advantages.detach().mean())
            advantage_abs_mean_list.append(advantages.detach().abs().mean())
            loss_scale_list.append(torch.tensor(loss_scale, device=device))
            sync_grad_list.append(torch.tensor(float(accelerator.sync_gradients), device=device))

        # Clean up KV caches (handles both t2i and it2i)
        from sensenova_u1.models.neo_unify.modeling_neo_chat import clear_flash_kv_cache
        for k in ("past_kv_cond", "past_kv_img_cond", "past_kv_uncond", "past_kv_tu"):
            cache = prefix_caches.get(k)
            if cache is not None:
                clear_flash_kv_cache(cache)

        # Aggregate metrics
        policy_loss = torch.stack(policy_loss_list).mean()
        policy_loss_abs = torch.stack(policy_loss_abs_list).mean()
        loss = torch.stack(loss_list).mean()
        loss_abs = torch.stack(loss_abs_list).mean()
        clipfrac = torch.stack(clipfrac_list).mean()
        clipfrac_gt_one = torch.stack(clipfrac_gt_one_list).mean()
        clipfrac_lt_one = torch.stack(clipfrac_lt_one_list).mean()
        kl_loss = torch.stack(kl_loss_list).mean()
        ratio_mean = torch.stack(ratio_mean_list).mean()
        ratio_std = torch.stack(ratio_std_list).mean()
        ratio_abs_deviation_mean = torch.stack(ratio_abs_deviation_mean_list).mean()
        old_logprob_mean = torch.stack(old_logprob_mean_list).mean()
        new_logprob_mean = torch.stack(new_logprob_mean_list).mean()
        logprob_delta_mean = torch.stack(logprob_delta_mean_list).mean()
        logprob_delta_abs_mean = torch.stack(logprob_delta_abs_mean_list).mean()
        advantage_mean = torch.stack(advantage_mean_list).mean()
        advantage_abs_mean = torch.stack(advantage_abs_mean_list).mean()
        loss_scale_mean = torch.stack(loss_scale_list).mean()
        sync_grad_fraction = torch.stack(sync_grad_list).mean()

        return {
            "clipfrac": clipfrac,
            "clipfrac_gt_one": clipfrac_gt_one,
            "clipfrac_lt_one": clipfrac_lt_one,
            "policy_loss": policy_loss,
            "policy_loss_abs": policy_loss_abs,
            "kl_loss": kl_loss,
            "loss": loss,
            "loss_abs": loss_abs,
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "ratio_abs_deviation_mean": ratio_abs_deviation_mean,
            "old_logprob_mean": old_logprob_mean,
            "new_logprob_mean": new_logprob_mean,
            "logprob_delta_mean": logprob_delta_mean,
            "logprob_delta_abs_mean": logprob_delta_abs_mean,
            "advantage_mean": advantage_mean,
            "advantage_abs_mean": advantage_abs_mean,
            "loss_scale": loss_scale_mean,
            "sync_grad_fraction": sync_grad_fraction,
            "grad_norm": _cur_grad_norm if isinstance(_cur_grad_norm, torch.Tensor) else torch.tensor(_cur_grad_norm),
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _compute_noise_scale(self, grid_h: int, grid_w: int) -> float:
        """Compute resolution-dependent noise scale (same logic as NEOChatModel)."""
        model = self.model
        noise_scale = model.noise_scale
        if model.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(model.noise_scale_base_image_seq_len)
            noise_scale = math.sqrt(
                (grid_h * grid_w) / (self.merge_size ** 2) / base
            ) * float(model.noise_scale)
            if model.noise_scale_mode == 'dynamic_sqrt':
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, model.noise_scale_max_value)
        return noise_scale

    def _decode_image(self, z: torch.Tensor, image_shapes: Tuple[int, int]) -> torch.Tensor:
        """Decode patchified z to [0,1] image tensor (C, H, W)."""
        image_prediction = self.model.unpatchify(
            z, self.patch_size * self.merge_size, image_shapes[1], image_shapes[0]
        )
        # Map from [-noise_scale, noise_scale] range to [0, 1]
        image = (image_prediction[0] * 0.5 + 0.5).clamp(0, 1).float()
        return image
