# Q-DETR: An Efficient Low-Bit Quantized Detection Transformer
在源码中根据论文添加了蒸馏部分的代码，如有疑问，欢迎交流！

# Train
PS: batch_size must be 2
```
python main.py --batch_size 2 --n_bit=4 --quant --finetune ./ckpt_4_bit_smca_detr_coco.pth \
--finetune_t ./ckpt_4_bit_smca_detr_coco.pth --num_queries 300 --coco_path your coco_path
```

# Usage
First, clone the repository locally:
```
git clone https://github.com/facebookresearch/detr.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Evaluation
Modifications in the script ```evaluate_coco.sh```:
* modify the **--coco_path** to your coco dataset
* define the **--n_bit** to **2**/**3**/**4**-bit 

```
bash evaluate_coco.sh
```

# Model Zoo
* Q-DETR based on SMCA-DETR on COCO

| Methods | Bit-width | Epoch | box AP |Model Link|
|:-------:|:---------:|:---------:|:--------------------:|:---:|
|Real-valued) |  32-bit     |  50     | 41.0 |-|
| [Q-DETR](https://arxiv.org/abs/2304.00253)    | 4-bit    |  50     | 38.5 |[Model](https://drive.google.com/file/d/1K_lZckXWW9_mdSvhXELeWoSyqvFZVJAT/view?usp=drive_link)|
| [Q-DETR](https://arxiv.org/abs/2304.00253)    | 2-bit    |  50     | 32.4 |[Model](https://drive.google.com/file/d/1sJ12U0s-df8YYLbAiPMI24p8qSOXBTk_/view?usp=drive_link)|

## Citation
If you find this repository useful, please consider citing our work:
```
@inproceedings{xu2023q,
  title={Q-DETR: An Efficient Low-Bit Quantized Detection Transformer},
  author={Xu, Sheng and Li, Yanjing and Lin, Mingbao and Gao, Peng and Guo, Guodong and L{\"u}, Jinhu and Zhang, Baochang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3842--3851},
  year={2023}
}
```

## Acknowledege
The project are borrowed heavily from Q-DETR.
