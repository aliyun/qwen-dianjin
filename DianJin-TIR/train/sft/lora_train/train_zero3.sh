export SWANLAB_LOG_DIR=./swanlab_log
export SWANLAB_MODE=local



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
accelerate launch --num_processes 8 --mixed_precision bf16 train.py --config sft_config.yaml
