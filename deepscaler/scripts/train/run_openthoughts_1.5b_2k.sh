#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Set default model path if not provided

MODEL_PATH="https://huggingface.co/google/gemma-3-1b-it"

PROJECT_NAME='openthoughts_8k_gemma1b'
EXPERIMENT_NAME='deepscaler-1.5b-2k' 

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_ratio=0.1 \
    +data.train_ratio_seed=42 \
    data.train_files=$HOME/data/openr1-math/train.parquet \
    data.val_files=[$HOME/data/aime/test.parquet,$HOME/data/amc/test.parquet,$HOME/data/math/test.parquet,$HOME/data/minerva/test.parquet] \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=21504 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    +actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.n_val=8 \
    +actor_rollout_ref.rollout.use_edit_for_validation=True \
    +actor_rollout_ref.rollout.use_longer_response_for_validation=True \
    actor_rollout_ref.rollout.force_append_answers=null \
    +actor_rollout_ref.rollout.forceans_for_untruncated=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="/mnt/disk3/verl/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.total_epochs=10 "${@:1}"\
    +reward_model.customized_reward_fn_name="deepscaler" \
    reward_model.edit_weight=0.0 \
    active_strategy.selection_metric=null \
    active_strategy.strategy_type=null
