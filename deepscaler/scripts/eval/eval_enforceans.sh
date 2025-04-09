set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Possible values: aime, amc, math, minerva, olympiad_bench
PROJECT_NAME='deepscaler_5k'
EXPERIMENT_NAME='deepscaler-1.5b-2k_forceans' 

# Check if command-line arguments are provided for n_pass and max_response_length
if [ "$#" -ge 1 ]; then
  N_PASS=$1
fi

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes

python3 -m verl.trainer.main_eval \
    data.path="/mnt/disk3/verl/eval/${PROJECT_NAME}/${EXPERIMENT_NAME}/gen.parquet" \
    data.response_key="responses" \
    data.edit_response_key="edit_responses" \
    data.n_pass=$N_PASS \
    data.max_response_length=2048 \
    +data.difficulty_key="difficulty" \
    +output_dir="/mnt/disk3/verl/eval/${PROJECT_NAME}/${EXPERIMENT_NAME}"
