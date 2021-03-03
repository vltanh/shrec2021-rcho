## Dataset preparation

### Download raw dataset

We will be working with the data for the Culture task as an example. The procedure is the same for the Shape task.

We are standing at the root directory of the project, or `.`

1. Make the directory `data`

```
$ mkdir data
$ cd data
```

2. Download and extract `datasetCulture.zip`

```
$ mkdir datasetShape
$ unzip datasetShape.zip -d datasetShape
```

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

1. Make a new directory called `list` in `./data/datasetCulture/`.

```
mkdir data/datasetCulture/list
```

2. Run

```
python scripts/gen_csv/cla2csv.py --input data/datasetCulture/dataset.cla --output data/datasetCulture/list/train.csv
```

This will generate a `train.csv` containing the object identifiers and the corresponding labels of the 3D objects in the collection (or gallery, train set, etc.).

**Example**:

| obj_id | class_id |
| ------ | -------- |
| 0      | 0        |
| 1      | 0        |
| 2      | 0        |

3. Run

```
python scripts/gen_csv/generate_test_csv.py --source data/datasetCulture/test --output data/datasetCulture/list/test.csv
```

This will generate a `test.csv` containing the object identifiers of the 3D objects in the query set (or test set).

**Example**:

| obj_id |
| ------ |
| 0      |
| 1      |
| 10     |

4. Run

```
python scripts/split_data/kfold_stratified_split.py --input data/datasetShape/list/train.csv --output data/datasetShape/list
```

This will generate multiple pairs of `<fid>_train.csv` and `<fid>_val.csv` files corresponding to the folds (`<fid>` stands for fold identifier). Each CSV is of the same format as the original `train.csv`.

By default, it is split into 5 folds. This can be changed.

### Ring data generation

The ring dataset has been generated beforehand and can be downloaded at

If you want to do it yourself then continue reading, else skip this section.

1. Download the blender tar file, version 2.79 (exclusive). Untar and check the `blender` executive inside.

```
$ wget https://download.blender.org/release/Blender2.79/blender-2.79-linux-glibc219-x86_64.tar.bz2
$ tar xjf blender-2.79-linux-glibc219-x86_64.tar.bz2
$ mv blender-2.79-linux-glibc219-x86_64.tar.bz2 blender-2.79
```

2. With `<phase>` being `train` or `test`, run

```
blender-2.79/blender -b -P scripts/gen_ring/generate_ring.py -- data/datasetCulture <phase>
```

It will create a checkpoint file `save.txt` as it runs, this file stores the index of the last generated object to resume in case of errors occuring. This file needs to be deleted between generation of each phase (else it will keep resuming).

A subdirectory `generated_<phase>` is created inside the `data/datasetCulture` folder. The structure looks like this:

```
data/
├─ datasetCulture/
│  ├─ generated_<phase>/
│  │  ├─ ring<rid>/
│  │  │  ├─ <type>/
│  │  │  │  ├─ Image<vid>.png
```

where

- `rid`: a number (default: 0-6) as ring identifier
- `type`: `depth`, `mask`, or `render`
- `view_id`: a number (with leading 0s, default: 0001-0012) as view identifier

## Train

## Extract

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
