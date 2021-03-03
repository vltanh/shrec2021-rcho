import os
import sys

ROOT = sys.argv[1]

def rename(path):
    ls = os.listdir(path)
    for fn in ls:
        if fn.endswith('.obj'):
            num = fn[:-4]
            num = num.zfill(4)
            new_fn = num + '.obj'
            os.rename(os.path.join(path, fn), os.path.join(path, new_fn))


for phase in ['train', 'test']:
    rename(os.path.join(ROOT, phase))
