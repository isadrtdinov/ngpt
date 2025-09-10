mycommand=" --init_from='scratch' --use_nGPT=1 --learning_rate=1e-3 --weight_decay=0.1 --warmup_iters=0"
torchrun --nnodes 1 --nproc_per_node 2 --rdzv_endpoint=localhost:29501 train.py $mycommand
