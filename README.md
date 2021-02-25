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
