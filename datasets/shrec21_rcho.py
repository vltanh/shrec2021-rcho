from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as tvtf

import csv
import os

__all__ = ['SHREC21_RCHO_Rings_RenderOnly']

class SHREC21_RCHO_Rings(data.Dataset):
    def __init__(self, 
                 csv_path, 
                 root, 
                 ring_ids, 
                 is_train=True):
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
                                     f'{int(obj_id):04d}', 
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
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.render_transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
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