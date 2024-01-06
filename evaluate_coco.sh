python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
--batch_size 2 --num_queries 300 --dynamic_scale type3 \
--quant --n_bit=x --finetune path_to/ckpt_x_bit_smca_detr_coco.pth --coco_path path_to/coco \
--eval 
