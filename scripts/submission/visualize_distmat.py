import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query',
                        type=str,
                        help='path to the query directory')
    parser.add_argument('-g', '--gallery',
                        type=str,
                        help='path to the gallery directory')
    parser.add_argument('-d', '--distmat',
                        type=str,
                        help='path to distance matrix txt')
    parser.add_argument('-o', '--out',
                        type=str,
                        help='path to output directory (must exist)')
    return parser.parse_args()


def load_3d_object(obj_id, root):
    im = Image.open(f'{root}/ring3/{obj_id:0>4}/render/Image0001.png')
    return im


def display_3d_object_on_ax(obj_id, root, ax):
    im = load_3d_object(obj_id, root)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])


args = parse_args()

dist_mtx = np.loadtxt(args.distmat)

for qid, qv in enumerate(dist_mtx):
    fig, axes = plt.subplots(1, 6)
    fig.set_size_inches(6, 1)

    display_3d_object_on_ax(qid, args.query, axes[0])
    axes[0].patch.set_edgecolor('blue')
    axes[0].patch.set_linewidth('5')

    first_5 = qv.argsort()[:5]
    for i, gid in enumerate(first_5):
        display_3d_object_on_ax(gid, args.gallery, axes[1+i])

    fig.tight_layout()
    plt.savefig(f'{args.out}/{qid}', dpi=200)
    plt.close()
