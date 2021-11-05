import argparse
import os
import torch
import tqdm
from fvcore.common.file_io import PathManager
# import pdb

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model
import numpy as np

from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

# A simple mapper from object detection dataset to VQA dataset names
dataset_to_folder_mapper = {}
dataset_to_folder_mapper['coco_2014_train'] = 'train2014'
dataset_to_folder_mapper['coco_2014_val'] = 'val2014'
# One may need to change the Detectron2 code to support coco_2015_test
# insert "coco_2015_test": ("coco/test2015", "coco/annotations/image_info_test2015.json"),
# at: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/builtin.py#L36
dataset_to_folder_mapper['coco_2014_test'] = 'test2014'

def extract_grid_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Grid feature extraction")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2014_train",
                        choices=['coco_2014_train', 'coco_2014_val', 'coco_2015_test'])
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def extract_grid_feature_on_dataset(model, data_loader, dump_folder):
    for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            image_id = inputs[0]['image_id']
            file_name = '%d.pth' % image_id
            # compute features
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            outputs = model.roi_heads.get_conv5_features(features)
            with PathManager.open(os.path.join(dump_folder, file_name), "wb") as f:
                # save as CPU tensors
                torch.save(outputs.cpu(), f)

def do_feature_extraction(cfg, model, dataset_name):
    with inference_context(model):
        dump_folder = os.path.join(cfg.OUTPUT_DIR, "features", dataset_to_folder_mapper[dataset_name])
        PathManager.mkdirs(dump_folder)
        data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)
        extract_grid_feature_on_dataset(model, data_loader, dump_folder)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # force the final residual block to have dilations 1
#     cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.WEIGHTS = 'output_X101/X-101.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # I do thresh filter in my code
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def resetup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # force the final residual block to have dilations 1
#     cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.WEIGHTS = 'output_X101/X-101.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0 # I do thresh filter in my code
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    do_feature_extraction(cfg, model, args.dataset)
    
    
    
    
args = extract_grid_feature_argument_parser().parse_args('--config-file configs/X-101-grid.yaml --dataset coco_2014_train'.split())


cfg = setup(args)
print(cfg)
model = build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    cfg.MODEL.WEIGHTS, resume=True
)

import h5py
import os
from detectron2.structures import Boxes


save_dir = '/home/luoyp/disk1/grid-feats-vqa/feats'
region_before = h5py.File(os.path.join(save_dir,'region_before_X101.hdf5'),'w')
# region_after = h5py.File(os.path.join(save_dir,'region_after.hdf5'),'w')
# grid7 = h5py.File(os.path.join(save_dir,'my_grid7.hdf5'),'w')
# original_grid = h5py.File(os.path.join(save_dir,'original_grid7.hdf5'),'w')

thresh = 0.2
max_regions = 100
pooling = torch.nn.AdaptiveAvgPool2d((7,7))
image_id_collector = []
for dataset_name in ['coco_2014_train','coco_2014_val']:
    with inference_context(model):
        dump_folder = os.path.join(cfg.OUTPUT_DIR, "features", dataset_to_folder_mapper[dataset_name])
        PathManager.mkdirs(dump_folder)
        data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)
        for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
            with torch.no_grad():
                image_id = inputs[0]['image_id']
                file_name = '%d.pth' % image_id
                images = model.preprocess_image(inputs)
                features = model.backbone(images.tensor)

                proposals, _ = model.proposal_generator(images, features)
                proposal_boxes = [x.proposal_boxes for x in proposals]

                features = [features[f] for f in model.roi_heads.in_features]
                box_features1 = model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
                box_features = model.roi_heads.box_head(box_features1)

                predictions = model.roi_heads.box_predictor(box_features)
                pred_instances, index = model.roi_heads.box_predictor.inference(predictions, proposals)


                topk = 10
                scores = pred_instances[0].get_fields()['scores']
                topk_index = index[0][:topk]

                thresh_mask = scores > thresh
                thresh_index = index[0][thresh_mask]

                if len(thresh_index) < 10:
                    index = [topk_index]
                elif len(thresh_index) > max_regions:
                    index = [thresh_index[:max_regions]]
                else:
                    index = [thresh_index]

                if len(topk_index) < 10:
                    print("{} has less than 10 regions!!!".format(image_id))
                    image_id_collector.append(image_id)
                    continue

                # feature of proposal
                proposal_box_features1 = box_features1[index].mean(dim=[2,3])
                proposal_box_features = box_features[index]
#                 pdb.set_trace()
                boxes = pred_instances[0].get_fields()['pred_boxes'].tensor[:len(index[0])]

                image_size = pred_instances[0].image_size

                assert boxes.shape[0] == proposal_box_features.shape[0]

                region_before.create_dataset('{}_features'.format(image_id),data=proposal_box_features1.cpu().numpy())
                region_before.create_dataset('{}_boxes'.format(image_id),data=boxes.cpu().numpy())
                region_before.create_dataset('{}_size'.format(image_id),data=np.array([image_size]))

#                 region_after.create_dataset('{}_features'.format(image_id),data=proposal_box_features.cpu().numpy())
#                 region_after.create_dataset('{}_boxes'.format(image_id),data=boxes.cpu().numpy())
#                 region_after.create_dataset('{}_size'.format(image_id),data=np.array([image_size]))
                


del cfg
del model

cfg = resetup(args)
print(cfg)
model = build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    cfg.MODEL.WEIGHTS, resume=True
)

print('problem images:')
print(image_id_collector)

for dataset_name in ['coco_2014_train','coco_2014_val']:
    with inference_context(model):
        dump_folder = os.path.join(cfg.OUTPUT_DIR, "features", dataset_to_folder_mapper[dataset_name])
        PathManager.mkdirs(dump_folder)
        data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)
        for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
            with torch.no_grad():
                image_id = inputs[0]['image_id']
                if image_id not in image_id_collector:
                    continue
                print('append image:',image_id)
                file_name = '%d.pth' % image_id
                images = model.preprocess_image(inputs)
                features = model.backbone(images.tensor)

                proposals, _ = model.proposal_generator(images, features)
                proposal_boxes = [x.proposal_boxes for x in proposals]

                features = [features[f] for f in model.roi_heads.in_features]
                box_features1 = model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
                box_features = model.roi_heads.box_head(box_features1)

                predictions = model.roi_heads.box_predictor(box_features)
                pred_instances, index = model.roi_heads.box_predictor.inference(predictions, proposals)
                
                topk = 10
                scores = pred_instances[0].get_fields()['scores']
                topk_index = index[0][:topk]

                thresh_mask = scores > thresh
                thresh_index = index[0][thresh_mask]
                
#                 if len(thresh_index) < 10:
                index = [topk_index]
                if len(topk_index) > max_regions:
                    index = [topk_index[:max_regions]]
                
                if len(topk_index) < 10:
                    print("{} has less than 10 regions!!!".format(image_id))
                    raise
    
                # feature of proposal
                proposal_box_features1 = box_features1[index].mean(dim=[2,3])
                proposal_box_features = box_features[index]

                boxes = pred_instances[0].get_fields()['pred_boxes'].tensor[:len(index[0])]
                image_size = pred_instances[0].image_size

                assert boxes.shape[0] == proposal_box_features.shape[0]

                region_before.create_dataset('{}_features'.format(image_id),data=proposal_box_features1.cpu().numpy())
                region_before.create_dataset('{}_boxes'.format(image_id),data=boxes.cpu().numpy())
                region_before.create_dataset('{}_size'.format(image_id),data=np.array([image_size]))

#                 region_after.create_dataset('{}_features'.format(image_id),data=proposal_box_features.cpu().numpy())
#                 region_after.create_dataset('{}_boxes'.format(image_id),data=boxes.cpu().numpy())
#                 region_after.create_dataset('{}_size'.format(image_id),data=np.array([image_size]))

region_before.close()
# region_after.close()
# grid7.close()
# original_grid.close()
