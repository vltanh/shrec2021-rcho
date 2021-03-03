import os
import glob
import csv
import sys

ROOT = sys.argv[1]

for phase in ['train', 'test']:
    OBJ_DIR = os.path.join(ROOT, phase)
    SAVE_DIR = os.path.join(ROOT, 'list')
    SAVE_FN = f'model_{phase}.txt'

    obj_ls = glob.glob(os.path.join(OBJ_DIR, '*.obj'))
    obj_ls = [[x.replace(ROOT, '')] for x in obj_ls]
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, SAVE_FN), 'w') as f:
        csv.writer(f).writerows(obj_ls)
