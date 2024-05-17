python test.py --num_process_per_node 8 --exp RFM_cifar10 \
        --dataset cifar10 --num_channel 128 --batch_size 125\
        --compute_fid --method dopri5 --reflect True\
        --output_log ./FID/RFM_cifar10.log 