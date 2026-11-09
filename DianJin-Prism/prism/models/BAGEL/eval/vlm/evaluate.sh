
set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

DATASET=${1}
echo "CHECKPOINT: ${CHECKPOINT}"

ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"

if  [ ${DATASET} == "mme" ]; then
  python -m eval.vlm.eval.mme.eval "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmvet" ]; then
    python -m eval.vlm.eval.mmvet.evaluate_mmvet --datasets mmvet "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-dev-en" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_dev_20230712 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-dev-cn" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_dev_cn_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-test-en" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_test_en_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-test-cn" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_test_cn_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-dev" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_dev "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-val" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_validation "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-val_cot" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu_cot --datasets MMMU_validation_cot "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-test" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_test "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_testmini "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_test "${ARGS[@]:1}"
fi

if [ ${DATASET} == "pope" ]; then
    torchrun \
    --nnodes=$ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m eval.vlm.eval.pope.evaluate_pope --datasets pope "${ARGS[@]:1}"
fi

if [ ${DATASET} == "pope_cot" ]; then
    torchrun \
    --nnodes=$ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m eval.vlm.eval.pope.evaluate_pope --datasets pope_cot --cot "${ARGS[@]:1}"
fi

if [ ${DATASET} == "vqa-gqa-testdev" ]; then
    torchrun \
    --nnodes=$ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m eval.vlm.eval.vqa.evaluate_vqa --datasets gqa_testdev_llava "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmvp" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmvp.evaluate_mmvp --datasets MMVP "${ARGS[@]:1}"
fi
