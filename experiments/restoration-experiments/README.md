## Image restoration experiments
This folder contains the scripts to run the image restoration experiments from the paper, including denoising and super-resolution.

### Training the restoration models

The `main.py` script is used to train the image restoration models using the STED-FM backbone. The training script allows for either training from scratch, only training a small linear layer on top of frozen features (linear probing), or fine-tuning the entire network. All of these configurations can also be performed in the small data regime. The examples below use the vit-small architecture with STED-FM weights. In essence, the training procedure is similar to the one used for segmentation fine-tuning (see `segmentation-experiments/README.md` for more details).

Below are example commands for training denoising and super-resolution models. Only the dataset name changes between the two tasks since the same training script is used.

```bash
python main.py --dataset <dataset_name> --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --use-tensorboard --opts "batch_size 32"
```

### Evaluating the restoration models

By default after training, the best model checkpoint (based on validation loss) is automatically evaluated on the test set. However, we also provide a separate evaluation script `eval.py` that can be used to evaluate the model. This script also provides the option to save example restored images from the test set.

```bash
python eval.py --dataset <dataset_name> --backbone mae-lightning-small --backbone-weights MAE_SMALL_STED --restore-from <path/to/checkpoint> --save-predictions --opts "batch_size 32"
```