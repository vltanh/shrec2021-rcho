from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np

import csv
import os

__all__ = ['SHREC21_RCHO_Rings_RenderOnly',
           'SHREC21_RCHO_Rings_RenderMask']


class SHREC21_RCHO_Rings(data.Dataset):
    def __init__(self,
                 csv_path,
                 root,
                 ring_ids,
                 is_train=True,
                 vis=False):
        csv_data = csv.reader(open(csv_path))
        next(csv_data)

        data, labels = list(zip(*csv_data))

        self.labels = list(map(int, labels))

        self.data = [
            {
                mode:
                [
                    [
                        os.path.join(root,
                                     f'ring{ring_id}',
                                     f'{obj_id:0>4}',
                                     mode,
                                     f'Image{view_id:04d}.png')
                        for view_id in range(1, 13)
                    ]
                    for ring_id in ring_ids
                ]
                for mode in ['depth', 'mask', 'render']
            }
            for obj_id in data
        ]

        if is_train:
            self.render_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])

            self.mask_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
            ])
        else:
            self.render_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])

            self.mask_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
            ])

        if vis:
            self.render_transforms = tvtf.Compose([
                tvtf.ToTensor(),
            ])

            self.mask_transforms = tvtf.Compose([
            ])

        self.is_train = is_train

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class SHREC21_RCHO_Rings_RenderOnly(SHREC21_RCHO_Rings):
    def __getitem__(self, i):
        data = self.data[i]['render']
        label = self.labels[i]

        ims = torch.cat([
            torch.cat([
                self.render_transforms(
                    Image.open(x).convert('RGB')
                ).unsqueeze(0)
                for x in views
            ]).unsqueeze(0)
            for views in data
        ])

        return(ims, label)


class SHREC21_RCHO_Rings_RenderMask(SHREC21_RCHO_Rings):
    def __getitem__(self, i):
        data = self.data[i]['render']
        masks = self.data[i]['mask']
        label = self.labels[i]

        ims = torch.cat([
            torch.cat([
                self.render_transforms(
                    Image.open(x).convert('RGB')
                ).unsqueeze(0)
                for x in views
            ]).unsqueeze(0)
            for views in data
        ])

        masks = torch.cat([
            torch.cat([
                torch.tensor(np.array(
                    self.mask_transforms(
                        Image.open(x).convert('L')
                    )) / 255.).long().unsqueeze(0).unsqueeze(0)
                for x in views
            ]).unsqueeze(0)
            for views in masks
        ])

        ims = torch.cat([ims, masks], 2)

        return (ims, label)


if __name__ == '__main__':
    ds = SHREC21_RCHO_Rings_RenderMask(
        'data/list/shape/1_train.csv', 'data/shape-train-rings', [0, 1])
    ds[0]
