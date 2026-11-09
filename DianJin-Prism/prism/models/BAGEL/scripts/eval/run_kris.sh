
set -x

export OPENAI_API_KEY=$openai_api_key

GPUS=8


torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_kris.py \
    --output_dir $output_path/bagel \
    --metadata_file ./eval/gen/kris/final_data.json \
    --max_latent_size 64 \
    --model-path $model_path \
    --think


python ./eval/gen/kris/metrics_common.py \
    --results_dir $output_path \
    --max_workers 8

python ./eval/gen/kris/metrics_knowledge.py \
    --results_dir $output_path \
    --max_workers 8

python ./eval/gen/kris/metrics_multi_element.py \
    --results_dir $output_path \
    --max_workers 8

python ./eval/gen/kris/metrics_temporal_prediction.py \
    --results_dir $output_path \
    --max_workers 8

python ./eval/gen/kris/metrics_view_change.py \
    --results_dir $output_path \
    --max_workers 8


python ./eval/gen/kris/summarize.py \
    --results_dir $output_path/bagel \