
if [ $# -lt 2 ]; then
    echo "Error: PREFIX_DIR and MODEL_PATH are required as the first and second arguments respectively."
    exit 1
fi

LOG_PATH=$1
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi
shift 1
ARGS=("$@")
export MASTER_PORT=10042

FULL_MODEL_PATH="$PREFIX_DIR/$MODEL_PATH"

IFS=' ' read -r -a DATASETS <<< "$DATASETS_STR"

for DATASET in "${DATASETS[@]}"; do
    bash eval/vlm/evaluate.sh \
        "$DATASET" \
        --out-dir "$LOG_PATH/$DATASET" \
        "${ARGS[@]}"
done