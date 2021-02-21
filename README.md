# Duel-Level Collaborative Transformer for Image Captioning
This repository contains the reference code for the paper [Duel-Level Collaborative Transformer for Image Captioning](https://arxiv.org/pdf/2101.06462.pdf).

![](https://raw.githubusercontent.com/luo3300612/image-captioning-DLCT/master/images/arch.png)

## Experiment setup
please refer to [m2 transformer](https://github.com/aimagelab/meshed-memory-transformer)

## Data preparation
* **Annotation**. Download the annotation file [annotation.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing)
* **Feature**. You can download our ResNeXt-101 feature (.hdf5 file) [here](https://pan.baidu.com/s/188xmv2r5eXUbEUqKSA4BCw). Access code: etrx.

There are five kinds of keys in our .hdf5 file. They are
* **Feature**. You can download our ResNeXt-101 feature [here](https://pan.baidu.com/s/188xmv2r5eXUbEUqKSA4BCw). Access code: etrx.

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

## Evaluation
```python
python eval.py --annotation annotation --workers 4 --features_path ./data/coco_all_align.hdf5 --model_path path_of_model_to_eval.pth --model DLCT --image_field ImageAllFieldWithMask --grid_embed --box_embed --dump_json gen_res.json --beam_size 5
```
Important args:
* `--features_path` path to hdf5 file
* `--model_path`
* `--dump_json` dump generated captions to


## Training

## References
[1] [M2](https://github.com/aimagelab/meshed-memory-transformer)

[2] [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)
## Acknowledgements
Thanks the original [m2](https://github.com/aimagelab/meshed-memory-transformer) and amazing work of [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa). 
