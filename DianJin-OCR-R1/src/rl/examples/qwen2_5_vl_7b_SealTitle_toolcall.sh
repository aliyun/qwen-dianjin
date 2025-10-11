#!/bin/bash
# bash examples/qwen2_5_vl_7b_SealTitle_toolcall.sh
set -x

export PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED=1
MODEL_PATH={{ your_model_path_here }} # e.g. Qwen2.5-VL-7B-Instruct
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.use_toolcall=True \
    data.train_files=$(pwd)/examples/demo_data/sealtitle_trainN64.parquet \
    data.val_files=$(pwd)/examples/demo_data/sealtitle_testN64.parquet \
    data.rollout_batch_size=32 \
    data.prompt_key=question \
    data.answer_key=transcription \
    data.format_prompt=$(pwd)/examples/format_prompt/seal_toolcall.jinja \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    data.max_pixels=1003520 \
    data.min_pixels=262144 \
    worker.rollout.n=8 \
    worker.rollout.max_num_batched_tokens=16900 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.reward.reward_function=$(pwd)/examples/reward_function/sealtitle.py:compute_score_withTool \
    trainer.experiment_name=qwen2_5_vl_7b_SealTitle_ToolCall\
    trainer.logger=["console"] \
    trainer.save_checkpoint_path=$(pwd)/cpkt_sealTitle_toolcall \
    trainer.total_epochs=5 \
    trainer.save_limit=-1 \
    trainer.save_freq=-1 \
    trainer.val_freq=2 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4