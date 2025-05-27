## Segmentation experiments
This folder contains the scripts for performing the supervised segmentation fine-tuning experiments as well as the patch retrieval experiment.

### Supervised segmentation fine-tuning

The fine-tuning script in `main.py` allows for either training from scratch, only training a small linear layer on top of frozen features (linear probing), or fine-tuning the entire network. All of these configurations can also be performed in the small data regime. The examples below use the vit-small architecture with STED-FM weights.

- For training from scratch:
```bash
python main.py --dataset <dataset_name> --backbone mae-lightning-small --backbone-weights None --opts "freeze_backbone false"
```

- For linear probing:
```bash
python main.py --dataset <dataset_name> --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --opts "freeze_backbone true"
```

- For end-to-end fine-tuning
```bash
python main.py --dataset <dataset_name> --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --opts "freeze_backbone false"
```

- The `--label-percentage` argument can be set to perform fine-tuning in the small data regime. For example, to run end-to-end fine-tuning on the factin dataset using only 1% of all available labels, one would run the following command:
```bash
python main.py --dataset factin --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --opts "freeze_backbone false" --label-percentage 0.01
```

### Patch retrieval
The steps to perform the patch retrieval experiment are outlined in the notebook `patch_retrieval.ipynb`.