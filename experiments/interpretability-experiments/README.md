## Attention maps experiment
This folder contains the script to extract attention maps from the different models using a given dataset, and saves the results for the ensuing user study.

To run the script using STED-FM weights:
```bash
python main_attentionmaps.py --dataset <dataset_name> --model mae-lightning-small --weights MAE_SMALL_STED
```