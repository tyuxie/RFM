python test.py --num_process_per_node 8 --exp RFM_ImageNet64_FM \
        --dataset ImageNet64 --num_channel 192 --cfg_scale 15.0 --batch_size 32 --num_steps 2  --method dopri5 \
        --compute_nfe --compute_fid  --output_log ./FID/RFM_dopri5_c192_cfg15p0.log --reflect True