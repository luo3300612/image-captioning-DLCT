# Dual-Level Collaborative Transformer for Image Captioning
This repository contains the reference code for the paper [Dual-Level Collaborative Transformer for Image Captioning](https://arxiv.org/pdf/2101.06462.pdf).

![](https://raw.githubusercontent.com/luo3300612/image-captioning-DLCT/master/images/arch.png)

## Experiment setup
please refer to [m2 transformer](https://github.com/aimagelab/meshed-memory-transformer)

## Data preparation
* **Annotation**. Download the annotation file [annotation.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing). Extarct and put it in the project root directory.
* **Feature**. You can download our ResNeXt-101 feature (hdf5 file) [here](https://pan.baidu.com/s/1Yeg7PNynhak7FL8aRLJ-Bw). Access code: 7fhb.
* **evaluation**. Download the evaluation tools [here](https://pan.baidu.com/s/1Yeg7PNynhak7FL8aRLJ-Bw). Access code: 7fhb. Extarct and put it in the project root directory.

There are five kinds of keys in our .hdf5 file. They are
* `['%d_features' % image_id]`: region features (N_regions, feature_dim)
* `['%d_boxes' % image_id]`: bounding box of region features (N_regions, 4)
* `['%d_size' % image_id]`: size of original image (for normalizing bounding box), (2,)
* `['%d_grids' % image_id]`: grid features (N_grids, feature_dim)
* `['%d_mask' % image_id]`: geometric alignment graph, (N_regions, N_grids)

We extract feature with the code in [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa).

The first three keys can be obtained when extracting region features with [extract_region_feature.py](./others/extract_region_feature.py).
The forth key can be obtained when extracting grid features with code in [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa).
The last key can be obtained with [align.ipynb](./align/align.ipynb)

## Training
```python
python train.py --exp_name dlct --batch_size 50 --head 8 --features_path ./data/coco_all_align.hdf5 --annotation annotation --workers 8 --rl_batch_size 100 --image_field ImageAllFieldWithMask --model DLCT --rl_at 17 --seed 118
```
## Evaluation
```python
python eval.py --annotation annotation --workers 4 --features_path ./data/coco_all_align.hdf5 --model_path path_of_model_to_eval --model DLCT --image_field ImageAllFieldWithMask --grid_embed --box_embed --dump_json gen_res.json --beam_size 5
```
Important args:
* `--features_path` path to hdf5 file
* `--model_path`
* `--dump_json` dump generated captions to

## References
[1] [M2](https://github.com/aimagelab/meshed-memory-transformer)

[2] [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)

[3] [butd](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1163.pdf)
## Acknowledgements
Thanks the original [m2](https://github.com/aimagelab/meshed-memory-transformer) and amazing work of [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa). 
