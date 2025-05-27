## Classification experiments.

This folder contains the code for doing KNN classification (`knn.py`) and supervised classification fine-tuning (`finetune.py`).  

The KNN classification script is straightforward.  

The fine-tuning script allows for either training from scratch, only training a small linear classifier on top of frozen features (linear probing), or fine-tuning the entire network. All of these configurations can also be performed in the small data regime. The examples below use the vit-small architecture with STED-FM weights.


- For training from scratch, the `--from scratch` argument must be passed to the script, e.g.  
    ```bash
    python finetune.py --from-scratch --dataset <dataset_name> --model mae-lightning-small --weights MAE_SMALL_STED
    ```
- For linear probing, the `blocks` argument must be set to `"all"`, e.g.  
    ```bash
    python finetune.py --blocks all --dataset <dataset_name> --model mae-lightning-small --weights MAE_SMALL_STED
    ```
- For full fine-tuning, the `blocks` argument must be set to 0, e.g.  
    ```bash
    python finetune.py --blocks 0 --dataset <dataset_name> --model mae-lightning-small --weights MAE_SMALL_STED  
    
- By default, the `num-per-class` argument is None so that all available training samples are used. For fine-tuning in the small data regime, this parameter must be set. Below is an example of full fine-tuning with the vit-tiny backbone pre-trained on the STED dataset, in the small data regime,  with 10 samples per class from the neural activity states dataset:  
    ```bash
    python finetune_v2.py --blocks 0 --num-per-class 10 --model mae-lightning-tiny --weights MAE_SMALL_STED --dataset neural-activity-states
    ```
