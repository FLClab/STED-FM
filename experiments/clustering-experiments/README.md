## Unsupervised experiments
This folder contains the scripts to run the recursive consensus clustering, image retrieval and patch retrieval experiments from the paper.

### Recursive Consensus Clustering (RCC)
The recursive consensus clustering procedure can be run on a dataset of images using either the deep (`deep` argument) features from STED-FM or hand-crafted (`manual` argument) features. Below is the example command to run RCC using the deep features.
```bash
python recursive_clustering.py --dataset <dataset_name> --mode <deep or manual>
``` 

Once the recursive clustering script has been run, the `analyze_clustering.py` script will create the clustering graphs color-coded by majority class proportion, and the `clustering_feature_graphs.py` will create the clustering graphs color-coded by handcrafted feature value. The `node_similarity.ipynb` notebook goes over the steps to evaluate the graphs and output the figures shown in the paper.

### Image retrieval
The image retrieval results can be reproduced by first running the following script which saves the results for all models on all datasets:
```bash
python image_retrieval.py 
```

The figures from the paper can then be reproduced using the following script:
```bash
python image_retrieval_results.py
```