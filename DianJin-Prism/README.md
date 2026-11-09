# Prism

Prism is a post-training framework for unified multimodal models. It generates
interleaved text and image trajectories, audits both sides, and creates targets
according to their reliability states.

## Install

Install a CUDA-compatible PyTorch and torchvision pair for the target machine
using the [PyTorch install selector](https://pytorch.org/get-started/locally/),
then install Prism:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -e .
```

`pip install -e .` installs the remaining training and evaluation dependencies.
FlashAttention is optional; install a wheel matching the installed CUDA, Python,
and PyTorch versions when available:

```bash
pip install flash-attn --no-build-isolation
```

The verified environment used PyTorch 2.9.0, Transformers 4.57.1, Accelerate
1.10.1, DeepSpeed 0.14.4, and PEFT 0.19.1.

## Download model weights

Model weights are not included. Download one supported checkpoint:

```bash
huggingface-cli download sensenova/SenseNova-U1-8B-MoT \
  --local-dir checkpoints/SenseNova-U1-8B-MoT

huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT \
  --local-dir checkpoints/BAGEL-7B-MoT
```

The bundled `data/train_prompts.jsonl` is the complete Prism training prompt
set and needs no additional download.

## Train

Run SenseNova-U1 training:

```bash
MODEL_PATH=checkpoints/SenseNova-U1-8B-MoT \
OUT=outputs/sensenova_online \
GPUS=0,1,2,3,4,5,6,7 PROCESSES=8 GRAD_ACCUM=8 \
SAVE_EVERY_STEPS=2 \
bash prism/shell/run_online.sh
```

Run BAGEL training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
  --num_processes 8 --config_file prism/configs/accelerate/zero2.yaml \
  prism/scripts/run_online.py --accelerate --adapter bagel \
  --model_path checkpoints/BAGEL-7B-MoT \
  --out outputs/bagel_online --grad_accum 8 --save_every_steps 2
```

## Export a checkpoint

After a training run saves `checkpoints/step22`, export that trained checkpoint:

```bash
RUN=outputs/sensenova_online
python -m prism.scripts.consolidate_ckpt \
  --ckpt "$RUN/checkpoints/step22" --out "$RUN/step22_lora.pt"
python -m prism.scripts.export_lora_hf \
  --base checkpoints/SenseNova-U1-8B-MoT \
  --lora "$RUN/step22_lora.pt" --r 64 --alpha 128 \
  --out outputs/eval_sensenova_step22
```

## Benchmarks

The included entrypoint supports six understanding benchmarks and two generation
benchmarks:

- Understanding: MME, MMBench, MMMU, MMStar, POPE, MMVP
- Generation: GenEval, DPG-Bench

Download every benchmark from its official project below. Prism does not modify
or redistribute benchmark data; place the official files into the following
layout before evaluation:

```text
benchmarks/
├── MME/test.jsonl
├── MMB/dev.jsonl
├── MMMU/validation.jsonl
├── MMStar/test.jsonl
├── POPE/test.jsonl
├── MMVP/test.jsonl
├── GenEval/prompts/evaluation_metadata.jsonl
└── DPGBench/prompts/
```

Each JSONL image field is resolved relative to its benchmark directory. Keep the
image files and any official companion files beside the downloaded JSONL.

Evaluate the exported training checkpoint:

```bash
MODEL_PATH=outputs/eval_sensenova_step22 \
BENCHMARK_ROOT=benchmarks \
OUT=outputs/eval_sensenova_step22 \
bash evaluation/run_benchmarks.sh
```

For a trained BAGEL checkpoint, set `MODEL=bagel` and use that exported
checkpoint as `MODEL_PATH`.

GenEval and DPG-Bench commands generate benchmark images. Run their official
scorers after generation. The six understanding runners score locally; MMMU
open-answer matching uses the official normalization logic embedded in its
runner.

| Benchmark | Official project |
|---|---|
| MME | [Evaluation source](https://github.com/bradyfu/awesome-multimodal-large-language-models/tree/Evaluation) |
| MMBench | [open-compass/MMBench](https://github.com/open-compass/MMBench) |
| MMMU | [MMMU-Benchmark/MMMU](https://github.com/MMMU-Benchmark/MMMU) |
| MMStar | [MMStar-Benchmark/MMStar](https://github.com/MMStar-Benchmark/MMStar) |
| POPE | [RUCAIBox/POPE](https://github.com/RUCAIBox/POPE) |
| MMVP | [tsb0601/MMVP](https://github.com/tsb0601/MMVP) |
| GenEval | [djghosh13/geneval](https://github.com/djghosh13/geneval) |
| DPG-Bench | [TencentQQGYLab/ELLA `dpg_bench`](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench) |

## License

Prism is MIT licensed. Bundled third-party source retains its upstream
license. Model weights are not included.
