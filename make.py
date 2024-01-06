import torch
from collections import OrderedDict

ckpt = torch.load('/home/b902-r4-01/code/SMCA-DETR/exps/backbone_quant_bit3_smca_single_scale_coco_50e/checkpoint.pth', map_location='cpu')

new = OrderedDict()

new['model'] = OrderedDict()

for k,v in ckpt['model'].items():
    if 'query_embed' in k:
        new['model'][k] = torch.cat([v, v], dim=1)
    else:
        new['model'][k] = v

torch.save(new, '/home/b902-r4-01/code/SMCA-DETR/exps/backbone_quant_bit3_smca_single_scale_coco_50e/ckpt_new.pth')