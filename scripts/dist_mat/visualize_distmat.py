import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query',
                        type=str,
                        help='path to the query directory')
    parser.add_argument('-g', '--gallery',
                        type=str,
                        help='path to the gallery directory')
    parser.add_argument('-c', '--csv',
                        type=str,
                        help='path to the gallery CSV')
    parser.add_argument('-d', '--distmat',
                        type=str,
                        help='path to distance matrix txt')
    parser.add_argument('-o', '--out',
                        type=str,
                        help='path to output directory')
    parser.add_argument('-k', '--k',
                        type=int,
                        help='number of retrieved items to visualize')
    parser.add_argument('-r', '--ring_ids',
                        nargs='+', type=int,
                        help='list of ring ids to use')
    return parser.parse_args()


def load_3d_object(obj_id, root, ring=3):
    im = Image.open(f'{root}/ring{ring}/{obj_id}/render/Image0001.png')
    return im


def display_3d_object_on_ax(obj_id, root, ax, label=None, ring=3):
    im = load_3d_object(obj_id, root, ring)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label)


args = parse_args()

dist_mtx = np.loadtxt(args.distmat)

df = csv.reader(open(args.csv))
next(df)
gid2label = {int(k): v for k, v in df}


if not os.path.exists(args.out):
    os.makedirs(args.out, exist_ok=True)

for qid, qv in enumerate(dist_mtx):
    k = min(args.k, len(qv))
    fig, axes = plt.subplots(len(args.ring_ids), 1+k)
    fig.set_size_inches(1+k, len(args.ring_ids))

    for i, rid in enumerate(args.ring_ids):
        display_3d_object_on_ax(qid, args.query, axes[i, 0], ring=rid)
        axes[i, 0].patch.set_edgecolor('blue')
        axes[i, 0].patch.set_linewidth('5')

    viz_items = qv.argsort()[:args.k]

    for j, rid in enumerate(args.ring_ids):
        for i, gid in enumerate(viz_items):
            display_3d_object_on_ax(
                gid, args.gallery, axes[j, 1+i],
                label=gid2label[gid],
                ring=rid
            )

    fig.tight_layout()
    plt.savefig(f'{args.out}/{qid}', dpi=200)
    plt.close()
