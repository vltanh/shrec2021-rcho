import argparse, os, csv
from sklearn.model_selection import StratifiedKFold
import pandas as pd
def parse_args():
    '''
    Parse arguments
    '''

    parser = argparse.ArgumentParser(description="Split the original dataset (`data.csv`) into two splits: train and val, from the given ratio.")
    parser.add_argument('-i', '--input', type=str, help="input csv file - csv_path")
    parser.add_argument('-o', '--output', type=str, help="path to the directory of the output csvs (default = ‘.’)",default=".")
    parser.add_argument('-k','--k_split',type=int,help="ratio of the train subset (default = 0.8, meaning is 5)",default=5)
    parser.add_argument('-s','--random_seed',type=int,help="seed for the randomization process (default = 0)",default=0)
    return parser.parse_args()
def write_to_csv(X_train,y_train,X_test,y_test,index,output):
    train = pd.DataFrame({"obj_id":X_train,"class_id":y_train})
    val = pd.DataFrame({"obj_id":X_test,"class_id":y_test})
    train.to_csv(os.path.join(output,str(index)+'_train.csv'),index=False)
    val.to_csv(os.path.join(output,str(index)+'_val.csv'),index=False)

def kfold_stratified_split(parser):
    '''
    To split the original dataset (`data.csv`) into k “folds”, each consists of 2 splits: train (n - n/k objects) and val (n/k objects).
    Arguments:
    `csv_path` (str): path to the `data.csv` file
    `out_dir` (str): path to the directory of the output csvs (default = ‘.’)
    `k` (int): ratio of the train subset (default = 0.8)
    `random_seed` (int): seed for the randomization process (default = 0)
    Results: write 2*k csv(s) `<i>_train.csv` and `<i>_val.csv` (i = fold identifier) to `out_dir`


    '''
    df = pd.read_csv(parser.input)
    y=df['class_id']
    X=df['obj_id']
    skf = StratifiedKFold(n_splits=parser.k_split,random_state=parser.random_seed,shuffle=True)
    i=1
    for train_index,val_index in skf.split(X,y):
        X_train,y_train = X[train_index],y[train_index]
        X_test,y_test = X[val_index],y[val_index]
        write_to_csv(X_train,y_train,X_test,y_test,i,parser.output)
        i=i+1
def main():
    args = parse_args()
    kfold_stratified_split(args)
main()
