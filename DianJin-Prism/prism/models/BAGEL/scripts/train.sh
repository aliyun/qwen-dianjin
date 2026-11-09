
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --llm_path $llm_path \
  --use_flex True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --max_latent_size 64  \
  --num_workers 1 # use small num_workers since the num_used_data (10) are not enough to split