#!/bin/bash
set -euo pipefail
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
BENCHMARK_ROOT=${BENCHMARK_ROOT:?set BENCHMARK_ROOT}
MODEL=${MODEL:-sensenova-u1}
OUT=${OUT:-outputs/evaluation}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
BENCHS=${BENCHS:-"mme mmbench mmmu mmstar pope mmvp geneval dpg"}
SENSENOVA_U1_RUNTIME=${SENSENOVA_U1_RUNTIME:-official}
export SENSENOVA_U1_RUNTIME
mkdir -p "$OUT"
for bench in $BENCHS; do
  case "$bench" in
    mme) python evaluation/benchmarks/eval_mme.py --model "$MODEL" --model_path "$MODEL_PATH" --data_dir "$BENCHMARK_ROOT/MME" --output "$OUT/mme.jsonl" --device "$DEVICE" --dtype "$DTYPE" ;;
    mmbench) python evaluation/benchmarks/eval_mmbench.py --model "$MODEL" --model_path "$MODEL_PATH" --data_dir "$BENCHMARK_ROOT/MMB" --output "$OUT/mmbench.jsonl" --device "$DEVICE" --dtype "$DTYPE" ;;
    mmmu) python evaluation/benchmarks/eval_mmmu.py --model "$MODEL" --model_path "$MODEL_PATH" --data_dir "$BENCHMARK_ROOT/MMMU" --output "$OUT/mmmu.jsonl" --device "$DEVICE" --dtype "$DTYPE" ;;
    mmstar) python evaluation/benchmarks/eval_mmstar.py --model "$MODEL" --model_path "$MODEL_PATH" --data_dir "$BENCHMARK_ROOT/MMStar" --output "$OUT/mmstar.jsonl" --device "$DEVICE" --dtype "$DTYPE" ;;
    pope) python evaluation/benchmarks/eval_pope.py --model "$MODEL" --model_path "$MODEL_PATH" --data_dir "$BENCHMARK_ROOT/POPE" --output "$OUT/pope.jsonl" --device "$DEVICE" --dtype "$DTYPE" ;;
    mmvp) python evaluation/benchmarks/eval_mmvp.py --model "$MODEL" --model_path "$MODEL_PATH" --data_dir "$BENCHMARK_ROOT/MMVP" --output "$OUT/mmvp.jsonl" --device "$DEVICE" --dtype "$DTYPE" ;;
    geneval) python evaluation/benchmarks/geneval_generate.py --model "$MODEL" --model_path "$MODEL_PATH" --prompts "$BENCHMARK_ROOT/GenEval/prompts/evaluation_metadata.jsonl" --outdir "$OUT/geneval" --device "$DEVICE" --dtype "$DTYPE" ;;
    dpg) python evaluation/benchmarks/dpg_generate.py --model "$MODEL" --model_path "$MODEL_PATH" --prompts_dir "$BENCHMARK_ROOT/DPGBench/prompts" --outdir "$OUT/dpg" --device "$DEVICE" --dtype "$DTYPE" ;;
    *) echo "unknown benchmark: $bench" >&2; exit 2 ;;
  esac
done
