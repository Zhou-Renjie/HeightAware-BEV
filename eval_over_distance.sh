#!/bin/bash

DATA_DIR="../nuscenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

MODEL_NAME="xxx"

EXP_NAME="HeightAware-BEV-eval-over-distance"

python eval_over_distance.py \
       --batch_size=1 \
       --exp_name=${EXP_NAME} \
       --dset='trainval' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_over_distance' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=2 \
       --ncams=6 \
       --encoder_type='res50' \
       --do_rgbcompress=True \
       --use_height_aware=True \
       --device_ids=[0]
