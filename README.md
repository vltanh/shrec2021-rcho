## Dataset preparation

We will be working with the data for the Shape task as an example. The procedure is the same for the Culture task.

First, make the directory `data` and extract `datasetShape.zip` into it.

The resulting directory tree should look like this:

```
./
├─ data/
│  ├─ datasetShape/
│  │  ├─ train/
│  │  ├─ test/
│  │  ├─ dataset.cla
├─ ...
```

Then, we generate the necessary CSV files and the ring dataset.

### CSV generation

1. Make a new directory called `list` in `./data/datasetShape`.

2. Run

```
python scripts/cla2csv.py --input data/datasetShape/dataset.cla --output data/datasetShape/list/train.csv
```

This will generate a `train.csv` containing the object identifiers and the corresponding labels of the 3D objects in the collection (or gallery, train set, etc.).

**Example**:

| obj_id | class_id |
| ------ | -------- |
| 79     | 0        |
| 80     | 0        |
| 81     | 0        |

3. Run

```
python scripts/generate_test_csv.py --source data/datasetShape/test --output data/datasetShape/list/test.csv
```

This will generate a `test.csv` containing the object identifiers of the 3D objects in the query set (or test set).

**Example**:

| obj_id |
| ------ |
| 0      |
| 1      |
| 2      |

4. Run

```
python scripts/kfold_stratified_split.py --input data/datasetShape/list/train.csv --output data/datasetShape/list
```

This will generate multiple pairs of `<fid>_train.csv` and `<fid>_val.csv` files corresponding to the folds (`<fid>` stands for fold identifier). Each CSV is of the same format as the original `train.csv`.

By default, it is split into 5 folds. This can be changed.

### Ring data generation

1. Run

```
    python generate_list.py data/datasetShape
```

This will generate a sub directory `list` in `data/datasetShape` which contains 2 files `model_{phase}.txt` which list the path to the `.obj` files in that corresponding phase.

Example:

```
model_train.txt
train/0158.obj
train/0219.obj
...
```

3. Download the blender tar file, version 2.79 (exclusive). Untar and check the `blender` executive inside.

4. Run

```
    /path/to/blender -b -P run.py -- data/dataset<task> <phase>
```

## Extract features

```
python extract.py \
 --weight backup/vanilla-f1/best_metric_F1.pth \
 --csv data/datasetShape/dataset.csv \
 --dir data/datasetShape/generated_train \
 --ring_ids 1 3 5 \
 --batch_size 16 \
 --output result/shape/embeddings \
 --id f1-vanilla-train \
 --gpu 0
```

## Generate distance matrix

```
python scripts/postprocess/gen_distmat.py \
  --query result/shape/embeddings/prob/f1-vanilla-train-prob.npy \
  --gallery result/shape/embeddings/prob/f1-vanilla-train-prob.npy \
  --mode dotprod \
  --output result/shape/dist_mtx/f1-vanilla-dotprod-train-train.txt
```

## Evaluate distance matrix

```
python scripts/evaluate/evaluate_distmat.py \
  --query data/datasetShape/list/${x}_val.csv \
  --gallery data/datasetShape/list/${x}_train.csv \
  --distmat result/shape/dist_mtx/f1-vanilla-dotprod-train-train.txt \
  --out report-f1-vanilla-dotprod.csv
```
