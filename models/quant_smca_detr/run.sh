#!/bin/sh
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_voc0712_cocofmt.py \
--quant \
--batch_size 4 --lr_drop 10 --num_queries 300 --epochs 12 --dynamic_scale type3 --output_dir exps/smca_single_scale_voc