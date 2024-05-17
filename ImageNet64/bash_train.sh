
#Note: we use fp32 training, which needs 8*8 A100 80G GPUs
python multi_node_env.py $(date +%G-%m-%d-%H-%M-%S) \
accelerate launch --multi_gpu --num_machines=%NNODES --num_processes=64 --machine_rank=%NODE_RANK --main_process_ip=%MASTER_ADDR --main_process_port=%MASTER_PORT \
    train.py --lr 1e-4 \
        --ema_decay 0.9999 \
        --batch_size 40 \
        --num_channel 192 \
        --total_steps 540001 \
        --save_step 20000 \
        --accm_grad 1\
        --warmup 5000\
        --exp './RFM_ImageNet64_FM'

