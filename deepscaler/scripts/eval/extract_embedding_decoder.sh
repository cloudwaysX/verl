set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME='deepscaler'
MODEL_NAME='deepseek-r1-distill-qwen-1.5b'
OUTPUT_DIR="/mnt/disk3/verl/embedding/${DATASET_NAME}/${MODEL_NAME}"


# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
python3 -m verl.trainer.main_inference \
    embedding.framework="hf" \
    embedding.n_gpus_per_node=8 \
    embedding.model.name=${MODEL_PATH} \
    data.path=$HOME/data/deepscaler/train.parquet \
    embedding.output_path=$OUTPUT_DIR \
    data.train_ratio=1 \











