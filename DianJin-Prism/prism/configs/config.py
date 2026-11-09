
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional, get_type_hints


@dataclass
class AdapterCfg:
    model: str = "sensenova_u1"
    model_path: str = "/path/to/SenseNova-U1-8B-MoT"
    repo_paths: dict[str, str] = field(
        default_factory=lambda: {
            "sensenova_u1": "",
        }
    )
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass
class SampleCfg:
    num_steps: int = 20
    cfg_scale: float = 4.0
    img_cfg_scale: float = 1.0
    timestep_shift: float = 3.0
    image_size: tuple[int, int] = (1024, 1024)
    think_mode: bool = False
    seed: int = 42
    max_images: int = 10


@dataclass
class AuditCfg:
    temperature: float = 0.7
    max_new_tokens: int = 4096
    retry_on_empty: bool = False
    think: bool = False
    think_cap_text: int = 1024
    think_cap_image: int = 1024
    finalizer_tokens: int = 4096
    verdict_provider: str = "self"
    external_max_new_tokens: int = 8192
    repair_tokens: int = 512
    answer_tries: int = 4


@dataclass
class TeachCfg:
    regen_num_steps: int = 50
    verify_iplus: bool = True
    verify_tplus: bool = True
    verify_max_retries: int = 3
    verify_temperature_step: float = 0.2


@dataclass
class LossCfg:
    unlike_clamp: float = 0.9
    cfg_txt_drop_prob: float = 0.1
    cfg_txtimg_drop_prob: float = 0.1


@dataclass
class TrainCfg:
    zero_stage: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    lr: float = 1e-4
    epochs: int = 1
    grad_accum: int = 1
    save_every_steps: int = 0
    keep_step_ckpts: int = 2
    save_freq: int = 1
    resume_from: str = ""
    seed: int = 42
    class_weights: dict = field(
        default_factory=lambda: {"s2t": 0.5, "t2i": 1.0, "i2t": 1.0, "neg": 0.005}
    )


@dataclass
class DataCfg:
    seeds: str = "data/seeds.jsonl"
    categories: list[str] = field(default_factory=lambda: ["tutorial", "reasoning", "story"])


@dataclass
class PrismConfig:
    mode: str = "offline"
    adapter: AdapterCfg = field(default_factory=AdapterCfg)
    sample: SampleCfg = field(default_factory=SampleCfg)
    audit: AuditCfg = field(default_factory=AuditCfg)
    teach: TeachCfg = field(default_factory=TeachCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
    output_dir: str = "outputs/prism"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PrismConfig":
        return _from_dict(cls, d)


def _from_dict(cls: type, d: Optional[dict]) -> Any:
    if d is None:
        return cls()
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)
