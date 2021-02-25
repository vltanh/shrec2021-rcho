import argparse
import os
from sklearn.model_selection import train_test_split
import pandas as pd


def parse_args():
    '''
    Parse arguments
    '''
    parser = argparse.ArgumentParser(
        description='Split the original dataset (`data.csv`) \
            into two splits: train and val, from the given ratio.'
    )
    parser.add_argument('-i', '--input', type=str,
                        help="input csv file - csv_path")
    parser.add_argument('-o', '--output', type=str, default=".",
                        help="path to the directory of the output csvs (default = ‘.’)")
    parser.add_argument('-r', '--raito', type=float, default=0.8,
                        help="ratio of the train subset (default = 0.8)")
    parser.add_argument('-s', '--random_seed', type=int, default=0,
                        help="seed for the randomization process (default = 0)")
    return parser.parse_args()


def stratified_split(parser):
    '''
    To split the original dataset (`data.csv`) into two splits: 
    train and val, from the given ratio.

    Arguments:
        `csv_path` (str): 
            path to the `data.csv` file
        `out_dir` (str): 
            path to the directory of the output csvs(default= ‘.’)
        `ratio` (float): 
            ratio of the train subset(default=0.8)
        `random_seed` (int): 
            seed for the randomization process(default=0)
    Results: 
        write two csv(s) `train.csv` and `val.csv` of 
        the same format as `data.csv` to `out_dir`
    '''
    df = pd.read_csv(parser.input)
    y = df['class_id']
    X = df['obj_id']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         train_size=parser.ratio,
                         random_state=parser.random_seed)
    train = pd.DataFrame({
        'obj_id': X_train,
        'class_id': y_train
    })
    val = pd.DataFrame({
        'obj_id': X_test,
        'class_id': y_test
    })
    train.to_csv(os.path.join(parser.output, 'train.csv'), index=False)
    val.to_csv(os.path.join(parser.output, 'val.csv'), index=False)


if __name__ == '__main__':
    args = parse_args()
    stratified_split(args)
