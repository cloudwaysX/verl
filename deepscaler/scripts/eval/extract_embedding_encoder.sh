set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="intfloat/e5-mistral-7b-instruct"
DATASET_NAME='openr1-math'
MODEL_NAME="e5-mistral-7b-instruct" 
OUTPUT_DIR="/mnt/disk3/verl/embedding/${DATASET_NAME}/${MODEL_NAME}"


# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
python3 -m verl.trainer.main_inference \
    embedding.framework="st" \
    embedding.n_gpus_per_node=8 \
    embedding.model.name=${MODEL_PATH} \
    data.path=$HOME/data/${DATASET_NAME}/train.parquet \
    embedding.output_path=$OUTPUT_DIR \
    data.train_ratio=1 \
    +model.encoder=True








