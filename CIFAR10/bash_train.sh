accelerate launch --num_processes 8\
    train.py --lr 2e-4 \
        --ema_decay 0.9999 \
        --batch_size 16 \
        --num_channel 128 \
        --total_steps 800001 \
        --save_step 20000 \
        --model fm\
        --exp './work_dir_TGaussian/RFM_cifar10'
