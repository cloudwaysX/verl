set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_unsft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_prompt_key=extra_info \
    data.train_response_key=extra_info\
    +data.train_prompt_dict_keys=['question'] \
    +data.train_response_dict_keys=['answer'] \
    data.val_prompt_key=extra_info \
    data.val_response_key=extra_info\
    +data.val_prompt_dict_keys=['question'] \
    +data.val_response_dict_keys=['answer'] \
    data.truncation="right" \
    optim.lr=1e-4 \
    data.micro_batch_size=1 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=evalconfidence_debug \
    trainer.experiment_name=gsm8k-qwen-2.5-0.5b-pretrain-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.total_training_steps=500 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
