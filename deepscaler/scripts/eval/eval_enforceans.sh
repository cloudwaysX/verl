set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Possible values: aime, amc, math, minerva, olympiad_bench
PROJECT_NAME='deepscaler_1k'
EXPERIMENT_NAME='deepscaler-1.5b-2k_foceans' 


# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.train_ratio=0.025 \
        +data.train_ratio_seed=42 \
        data.path=$HOME/data/deepscaler/train.parquet \
        data.output_path="/mnt/disk3/verl/eval/${PROJECT_NAME}/${EXPERIMENT_NAME}/gen.parquet" \
        data.n_samples=8 \
        data.batch_size=1024 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=2048 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1 \
        rollout.force_append_answers=True

    python3 -m verl.trainer.main_eval \
        data.path="/mnt/disk3/verl/eval/${PROJECT_NAME}/${EXPERIMENT_NAME}/gen.parquet" \
        data.response_key="response" \
        data.edit_response_key="edit_response" 
