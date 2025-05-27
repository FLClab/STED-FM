## Image generation experiments 
This folder contains the scripts to run the image generation and latent attribute manipulation experiments from the paper. 

### DDPM training 
The `train_latent_guidance.py` script trains the DDPM conditioned on STED-FM's latent vectors using the 240k subset of the large scale STED dataset.  
The `train_classifier_guidance.py` trains the DDPM conditioned on class embeddings (classifier-free guidance) using the 240k subset of the large scale STED dataset.

### Latent attribute manipulation
The latent attribute manipulation experiments are performed in 3 main steps. Below are exemples for the synaptic development dataset.

1) The dataset of images first needs to be embedded in the latent space of STED-FM.
```bash
python embed_dataset.py --dataset als --split train 
python embed_dataset.py --dataset als --split valid
```

2) A SVM is the trained to classify the embeddings. In the case of the synaptic development dataset, the classes are DIV12 and DIV25. The script saves the SVM's decision boundary for the next steps.

```bash
python train_boundary.py --dataset als 
```

3.1) First, the distribution of the train images' distance to the decision boundary is computed using the script below with the `--sanity-check` argument. 

```bash
python als_experiment.py --sanity-check
```

3.2) The latent attribute manipulation can now be performed with the following:
```bash
python als_experiment.py
```

### Other experiments
All experiments can be performed by doing steps 1-2 with the adequate arguments. For step 3, the experiments are performed by different scripts.  
- *actin to tubulin experiment*: `tubulin_actin_experiment.py`
- 

