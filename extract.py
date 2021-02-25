import os
import csv
import argparse

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tvtf
from tqdm import tqdm
from PIL import Image
import numpy as np

from torchan.utils.getter import get_instance
from torchan.utils.device import move_to, detach


class SHREC21_RCHO_RingsTest:
    def __init__(self,
                 csv_path,
                 root,
                 ring_ids,
                 use_mask=False):
        csv_data = csv.reader(open(csv_path))
        next(csv_data)
        data = [x[0] for x in csv_data]

        self.data = [
            self.get_by_id(obj_id, root, ring_ids)
            for obj_id in data
        ]

        self.render_transforms = tvtf.Compose([
            tvtf.CenterCrop((352, 352)),
            tvtf.Resize((224, 224)),
            tvtf.ToTensor(),
            tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        ])

        self.use_mask = use_mask
        if self.use_mask:
            self.mask_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
            ])

    def get_by_id(self, obj_id, root, ring_ids):
        d = {
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
        d['id'] = obj_id
        return d

    def __getitem__(self, i):
        data = self.data[i]['render']
        obj_id = self.data[i]['id']

        ims = torch.cat([
            torch.cat([
                self.render_transforms(
                    Image.open(x).convert('RGB')
                ).unsqueeze(0)
                for x in views
            ]).unsqueeze(0)
            for views in data
        ])

        if self.use_mask:
            masks = self.data[i]['mask']
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

        return ims, obj_id

    def __len__(self):
        return len(self.data)


def invert(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight',
                        type=str,
                        help='path to weight files')
    parser.add_argument('-c', '--csv',
                        type=str,
                        help='path to test csv file')
    parser.add_argument('-d', '--dir',
                        type=str,
                        help='path to the ring views directory')
    parser.add_argument('-rids', '--ring_ids',
                        nargs='+', type=int,
                        help='path to the ring views directory')
    parser.add_argument('-b', '--batch_size',
                        type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('-g', '--gpu',
                        type=int, default=None,
                        help='(single) GPU to use (default: None)')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='output directory')
    parser.add_argument('-i', '--id',
                        type=str,
                        help='model id')
    parser.add_argument('-m', '--mask',
                        action='store_true',
                        help='use mask')
    parser.add_argument('-n', '--num_workers',
                        type=int,
                        default=8,
                        help='number of CPU workers')
    return parser.parse_args()


def generate_device(gpu):
    dev_id = 'cuda:{}'.format(gpu) \
        if torch.cuda.is_available() and gpu is not None \
        else 'cpu'
    device = torch.device(dev_id)
    return dev_id, device


def generate_model(pretrained, dev_id, device):
    model_cfg = torch.load(pretrained, map_location=dev_id)
    model = get_instance(model_cfg['config']['model']).to(device)
    model.load_state_dict(model_cfg['model_state_dict'])
    return model


args = parse_args()

# Device
dev_id, device = generate_device(args.gpu)

# Load model
model = generate_model(args.weight, dev_id, device)

# Load data
dataset = SHREC21_RCHO_RingsTest(csv_path=args.csv,
                                 root=args.dir,
                                 ring_ids=args.ring_ids,
                                 use_mask=args.mask)
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)

# Extract embeddings
embeddings = {
    'feature': [],
    'logit': [],
    'prob': [],
}
ids = []
with torch.no_grad():
    model.eval()
    progress_bar = tqdm(dataloader)
    for i, (inp, lbl) in enumerate(progress_bar):
        # Load inputs and labels
        inp = move_to(inp, device)

        # Get network outputs
        features = model.get_embedding(inp)
        logits = model.get_logit_from_emb(features)
        probs = torch.softmax(logits, dim=1)

        print(torch.softmax(model(inp)[:, -1], dim=1) - probs)

        embeddings['feature'].append(detach(features).cpu())
        embeddings['logit'].append(detach(logits).cpu())
        embeddings['prob'].append(detach(probs).cpu())

        # Get label
        ids += lbl

ids = list(map(int, ids))

for m in embeddings.keys():
    embeddings[m] = torch.cat(embeddings[m], dim=0).numpy()

    # Rearrange order
    embeddings[m] = embeddings[m][invert(ids)]

    # Save result
    np.save(f'{args.output}/{m}/{args.id}-{m}.npy', embeddings[m])
