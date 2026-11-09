
from __future__ import annotations

from typing import Any, Iterable, Optional

from .objective import compute_loss
from ..core.types import TeachSample

_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen",
    "gate_proj", "up_proj", "down_proj",
]

_LORA_TARGETS_BAGEL = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen",
    "gate_proj", "up_proj", "down_proj",
]


def _select_lora_targets(is_bagel: bool) -> list[str]:
    return _LORA_TARGETS_BAGEL if is_bagel else _LORA_TARGETS


class PrismTrainer:
    def __init__(self, adapter: Any, cfg: Any, accelerator: Any = None) -> None:
        import torch

        self.torch = torch
        self.adapter = adapter
        self.cfg = cfg
        self.acc = accelerator
        self._setup_trainable()

    def _setup_trainable(self) -> None:
        torch = self.torch
        model = self.adapter.model
        is_bagel = any("_moe_gen" in n for n, _ in model.named_parameters())
        lora_targets = _select_lora_targets(is_bagel)

        from peft import LoraConfig, get_peft_model

        model.language_model = get_peft_model(
            model.language_model,
            LoraConfig(
                r=self.cfg.train.lora_rank, lora_alpha=self.cfg.train.lora_alpha,
                target_modules=lora_targets,
                lora_dropout=0.0, bias="none",
            ),
        )
        if is_bagel:
            base_lm = getattr(model.language_model, "model", None)
            inner = getattr(base_lm, "model", None)
            if base_lm is not None and inner is not None and hasattr(inner, "embed_tokens"):
                class _EmbedAlias:
                    def __init__(self, emb):
                        self._emb = emb

                    def __call__(self, input_ids):
                        return self._emb(input_ids)

                base_lm.embed_tokens = _EmbedAlias(inner.embed_tokens)
        if self.acc is None:
            for n, p in model.named_parameters():
                if "fm_modules" in n:
                    p.requires_grad = True

        if self.acc is not None:
            lm_ids = {id(p) for p in model.language_model.parameters()}
            for p in model.parameters():
                if id(p) not in lm_ids:
                    p.requires_grad = False

        import os as _os
        if _os.environ.get("PRISM_GRAD_CKPT") == "1":
            if is_bagel:
                if self._is_main():
                    print("[trainer] gradient checkpointing delegated to BagelAdapter")
            else:
                try:
                    model.language_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                    if hasattr(model.language_model, "config"):
                        model.language_model.config.use_cache = False
                    if self._is_main():
                        print("[trainer] gradient checkpointing ON (use_reentrant=False)")
                except Exception as e:
                    if self._is_main():
                        print(f"[trainer] WARN gradient checkpointing unavailable: {e}")

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.cfg.train.lr)

        if self.acc is not None:
            dsp = getattr(self.acc.state, "deepspeed_plugin", None)
            if dsp is not None:
                dsp.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
                dsp.deepspeed_config["gradient_accumulation_steps"] = self.cfg.train.grad_accum
            model.language_model, self.opt = self.acc.prepare(model.language_model, opt)
        else:
            self.opt = opt
        self._micro = 0
        self.params = params

        n = sum(p.numel() for p in params)
        if self._is_main():
            print(f"[trainer] lora_targets={lora_targets} accel={'yes' if self.acc else 'no'} "
                  f"trainable={n/1e6:.1f}M")

    def _is_main(self) -> bool:
        return self.acc is None or self.acc.is_main_process

    def _cur_lr(self) -> float:
        try:
            opt = getattr(self.opt, "optimizer", self.opt)
            return float(opt.param_groups[0]["lr"])
        except Exception:
            return float(self.cfg.train.lr)

    def _lm_train_mode(self, on: bool) -> None:
        try:
            lm = self.adapter.model.language_model
            lm.train() if on else lm.eval()
        except Exception:
            pass

    def train_step(self, sample: TeachSample) -> Optional[float]:
        ga = max(1, self.cfg.train.grad_accum)
        self.last_grad_norm = None
        self.last_stepped = False
        self._lm_train_mode(True)
        if self.acc is not None:
            with self.acc.accumulate(self.adapter.model.language_model):
                loss = compute_loss(self.adapter, sample, self.cfg)
                if loss is None:
                    self._lm_train_mode(False)
                    return None
                self.acc.backward(loss)
                self.last_stepped = bool(self.acc.sync_gradients)
                if self.acc.sync_gradients:
                    gn = self.acc.clip_grad_norm_(self.params, 1.0)
                    self.last_grad_norm = float(gn) if gn is not None else None
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
            self.last_lr = self._cur_lr()
            self._lm_train_mode(False)
            return float(loss.detach())
        loss = compute_loss(self.adapter, sample, self.cfg)
        if loss is None:
            self._lm_train_mode(False)
            return None
        (loss / ga).backward()
        self._micro += 1
        self.last_stepped = (self._micro % ga == 0)
        if self.last_stepped:
            gn = self.torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.last_grad_norm = float(gn)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        self.last_lr = self._cur_lr()
        self._lm_train_mode(False)
        return float(loss.detach())

    def train(self, samples: Iterable[TeachSample]) -> list[float]:
        samples = list(samples)
        losses = []
        for epoch in range(self.cfg.train.epochs):
            for i, s in enumerate(samples):
                lv = self.train_step(s)
                if lv is not None:
                    losses.append(lv)
                    if self._is_main():
                        print(f"[trainer] epoch={epoch} step={i} kind={s.kind} loss={lv:.5f}")
        return losses

    def save(self, path: str) -> None:
        import os

        if not self._is_main():
            return
        os.makedirs(path, exist_ok=True)
        model = self.adapter.model
        lm = self.acc.unwrap_model(model.language_model) if self.acc is not None else model.language_model
        lm.save_pretrained(path)
        extra = {n: p.detach().cpu() for n, p in model.named_parameters()
                 if p.requires_grad and "fm_modules" in n}
        if extra:
            self.torch.save(extra, os.path.join(path, "prism_gen_extra.pt"))
        print(f"[trainer] saved -> {path}")

    def load(self, path: str) -> None:
        import os

        model = self.adapter.model
        lm = self.acc.unwrap_model(model.language_model) if self.acc is not None else model.language_model
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        ckpt = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt}")
        set_peft_model_state_dict(lm, load_file(ckpt))
        extra_p = os.path.join(path, "prism_gen_extra.pt")
        if os.path.exists(extra_p):
            model.load_state_dict(self.torch.load(extra_p, map_location="cpu"), strict=False)
        if self._is_main():
            print(f"[trainer] loaded <- {path}")

    def save_state(self, path: str) -> None:
        import os

        if self.acc is not None:
            self.acc.save_state(path)
            return
        if not self._is_main():
            return
        os.makedirs(path, exist_ok=True)
        model = self.adapter.model
        sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        self.torch.save({"model": sd, "optim": self.opt.state_dict(), "micro": self._micro},
                        os.path.join(path, "trainer_state.pt"))
        print(f"[trainer] state saved -> {path}")

    def load_state(self, path: str) -> None:
        import os

        if self.acc is not None:
            torch = self.torch
            _orig = torch.load

            def _load(*a, **kw):
                kw["weights_only"] = False
                return _orig(*a, **kw)

            torch.load = _load
            try:
                self.acc.load_state(path)
            finally:
                torch.load = _orig
        else:
            st = self.torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu")
            self.adapter.model.load_state_dict(st["model"], strict=False)
            self.opt.load_state_dict(st["optim"])
            self._micro = int(st.get("micro", 0))
        if self._is_main():
            print(f"[trainer] state loaded <- {path}")
