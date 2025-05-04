python analyze_var_sim.py \
    --data_path=$HOME/data/deepscaler/train.parquet \
    --embedding_path=/mnt/disk3/verl/embedding/deepscaler/e5-mistral-7b-instruct/embeddings.npy \
    --variance_path=/mnt/disk3/verl/eval/deepscaler_1k/deepscaler-1.5b-8k_forceans/variances.json \
    --train_ratio=0.025 \
    --train_ratio_seed=42 \
    --n_pairs=60000
