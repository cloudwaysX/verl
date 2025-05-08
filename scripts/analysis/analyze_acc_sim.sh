python analyze_acc_sim.py \
    --data_path=$HOME/data/deepscaler/train.parquet \
    --embedding_path=/mnt/disk3/verl/embedding/deepscaler/e5-mistral-7b-instruct/embeddings.npy \
    --acc_path=/mnt/disk3/verl/eval/deepscaler_5k/deepscaler-1.5b-2k_forceans/mean_accs.json \
    --train_ratio=0.125 \
    --train_ratio_seed=42 \
    --n_pairs=60000
