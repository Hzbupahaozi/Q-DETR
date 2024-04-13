python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
--batch_size 2 --num_queries 300 --dynamic_scale type3 \
--quant --n_bit=4 --finetune ./ckpt_4_bit_smca_detr_coco.pth --coco_path 'E:/dataset/COCO' \
--eval 
