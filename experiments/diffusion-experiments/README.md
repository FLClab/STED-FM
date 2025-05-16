## Image generation experiments 
Emdedding the dataset

```bash
python embed_dataset.py --dataset activity --split train 
python embed_dataset.py --dataset activity --split valid
```

Training the SVM model

```bash
python train_boundary.py --dataset activity 
```

Evaluating the model

```bash
python activity_experiment.py --boundary activity
```

