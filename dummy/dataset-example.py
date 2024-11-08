
import sys
sys.path.insert(0, "../experiments/")

from datasets import get_dataset

dataset = get_dataset(
    "STED", 
    "/home/anthony/Documents/flc-dataset/experiments/simclr-experiments/data/FLCDataset/dataset.tar",
    use_cache=False, debug=True, 
    return_metadata=True)

print(dataset[0])