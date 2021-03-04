## Dataset

- RAW dataset:
  - Shape: [GGDrive](https://drive.google.com/file/d/1E38j-iopOMMzpaRwCRwDKXZpzRptrMga/view)
  - Culture: [GGDrive](https://drive.google.com/file/d/1rxmMABISRWcNqWNWwKnzH6njdtdcwg0v/view)
- Generated RingView: [GGDrive](https://drive.google.com/drive/folders/1RNYVodRlUubzQB4IB1bPMe8-NxFEvGfi?usp=sharing)

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

The ring dataset has been generated and included above, if you want to do it yourself then continue reading, else skip this section.

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

### Config

The training use a YAML file as configuration. Check `configs/train` for some examples.

### Initiate

To start training, run

```
python train.py --config configs/train/culture-vanilla.yaml --gpus 0
```

where:

- `config`: path to the configuration file
- `gpus`: specify which gpu(s) to use (currently can only handle 1 gpu)

### Monitor

To monitor the training process, run

```
tensorboard --logdir runs --port 6006
```

where:

- `logdir`: path to logging directory (`runs` by default, can be changed in configuration file)
- `port`: port to expose the visualization web app

Keep it running and go to `localhost:6006` on your browser.

### Evaluate

To evaluate a trained model, run

```
python val.py --config configs/val/culture-vanilla.yaml --gpus 0
```

where:

- `config`: path to the configuration file
- `gpus`: specify which gpu(s) to use (currently can only handle 1 gpu)

The result will be a on-screen report of the given metrics.

## Feature Extraction

To extract features using the pretrained model, run

```
python extract.py \
 --weight backup/culture-masked-f1/best_metric_F1.pth \
 --csv data/datasetCulture/train.csv \
 --dir data/datasetCulture/generated_train \
 --ring_ids 1 3 5 \
 --mask \
 --batch_size 16 \
 --output result/culture/embeddings \
 --id culture-f1-masked-train \
 --gpu 0
```

where:

- `--weight`: path to the pretrained weight
- `--csv`: path to a CSV file containing an `obj_id` column with object identifiers, these will be the objects in request of extraction
- `--dir`: path to the directory containing rings data
- `--ring_ids`: list of rings to use
- `--mask`: if the model needs to use binary mask
- `--batch_size`: batch size of the data loading
- `--output`: output director of the extraction (will be created if is not already existed)
- `--id`: identifier for this model
- `--gpu`: which gpu to use

The result will be a directory containing `.npy` files. It should look like this.

```
<output>/
├─ feature/
│  ├─ <id>-feature.npy
├─ prob/
│  ├─ <id>-prob.npy
```

Each `.npy` files is a $N \times D$ matrix, where $N$ is the number of objects and $D$ is the dimension of the feature vector.

## Distance matrix

## Generation

Given two embedding matrices (a query can be regarded as a matrix with 1 row), we can generate a distance matrix by running:

```
python scripts/postprocess/gen_distmat.py \
  --query result/culture/embeddings/prob/culture-f1-masked-train-prob.npy \
  --gallery result/shape/embeddings/prob/culture-f1-masked-train-prob.npy \
  --mode dotprod \
  --output result/shape/dist_mtx/f1-masked-dotprod-train-train.txt \
  --format %10.8f
```

where:

- `query`: path to the embedding matrix of the query
- `gallery`: path to the embedding matrix of the gallery
- `mode`: distance metric to be used [dotprod|euclidean|cosine]
- `output`: filename and directory of the output file
- `format`: floating point format

The output is a $Nq \times Ng$ matrix $M$ where $Nq, Ng$ are respectively the number of objects in the query and the gallery. It is saved in a `.txt` file.

## Evaluation

Given a distance matrix and two lists of objects with its corresponding category label, we can calculate the retrieval scores by running

```
python scripts/evaluate/evaluate_distmat.py \
  --query data/datasetShape/list/1_val.csv \
  --gallery data/datasetShape/list/2_train.csv \
  --distmat result/shape/dist_mtx/culture-f1-masked-dotprod-train-train.txt \
  --out report-f1-masked-dotprod.csv
```

where:

- `query`: path to the CSV file containing `obj_id, class_id`
- `gallery`: same as `query` but for the gallery
- `distmat`: path to the distance matrix `.txt` file
- `output`: filename and directory of the CSV report

The given 2 lists will provide the indices of interest for the row (query) and the column (gallery) of the distance matrix.

The result will be an on-screen summary report and a CSV report on each query.

**Example** (on-screen report)

```
MAP    0.834007
NN     0.820225
FT     0.794053
ST     1.180526

               MAP       NN        FT        ST
class_id
0         0.541112  0.50000  0.542857  0.739286
1         0.969604  0.96875  0.942034  1.420613
2         0.010932  0.00000  0.000000  0.000000
3         0.071429  0.00000  0.000000  0.000000
4         0.525223  0.50000  0.341667  0.529167
5         0.707407  0.60000  0.660870  0.800000
```

**Example** (CSV report)

| obj_id | class_id | MAP                  | NN  | FT                 | ST  |
| ------ | -------- | -------------------- | --- | ------------------ | --- |
| 3      | 0        | 0.2                  | 0   | 0.8823529411764706 | 1.0 |
| 13     | 0        | 0.00390625           | 0   | 0.0                | 0.0 |
| 20     | 0        | 0.003484320557491289 | 0   | 0.0                | 0.0 |
