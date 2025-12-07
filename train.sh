#!/bin/bash

DATA_DIR="../nuscenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

EXP_NAME="HeightAware-BEV"

python train_nuscenes.py \
       --exp_name=${EXP_NAME} \
       --max_iters=40000 \
       --log_freq=2000 \
       --save_freq=5000 \
       --dset='trainval' \
       --batch_size=16 \
       --grad_acc=2 \
       --data_dir=$DATA_DIR \
       --log_dir='logs_nuscenes' \
       --ckpt_dir='checkpoints' \
       --res_scale=2 \
       --ncams=6 \
       --encoder_type='res50' \
       --do_rgbcompress=True \
       --use_height_aware=True \
       --device_ids=[0,1,2,3,4,5,6,7]

