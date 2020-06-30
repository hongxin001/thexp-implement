import os

from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Subset
import random

# TODO 实现有些麻烦，暂时使用其他复现的mnist
class Wrap:
    def cls_dict(self):
        raise NotImplementedError()

    def initial(self, datasize=5000, proportions=None, val_per_cls=5, *clss):
        cls_dict = self.cls_dict()
        data_idx = []
        val_idx = []
        cls_map = {}
        for new_cls, cls in enumerate(clss):
            data_idx.extend(cls_dict[cls])
            val_idx.extend(random.sample(cls_dict[cls], val_per_cls))
            cls_map[cls] = new_cls
        self.data_idx = data_idx
        self.val_idx = val_idx
        self.cls_map = cls_map

    def to_newcls(self, cls):
        return self.cls_map[cls]


class MNISTImblance(MNIST, Wrap):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.initial()

    @property
    def raw_folder(self):
        return os.path.join(self.root, "MNIST", 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, "MNIST", 'processed')
