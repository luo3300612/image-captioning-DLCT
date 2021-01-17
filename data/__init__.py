from .field import *
from .dataset import COCO
from torch.utils.data import DataLoader as TorchDataLoader


class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)


def build_image_field(args):
    if args.image_field == 'ImageAllFieldWithMask':
        return ImageAllFieldWithMask(detections_path=args.features_path,
                             max_detections=args.max_detections,
                             load_in_tmp=False)
    else:
        raise NotImplementedError('No field: {}'.format(args.image_field))
