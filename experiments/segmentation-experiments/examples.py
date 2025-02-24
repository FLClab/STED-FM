
import random
import numpy

from datasets import get_dataset
from matplotlib import pyplot

from tiffwrapper import make_composite

import sys 
sys.path.insert(0, "..")
from model_builder import get_base_model, get_pretrained_model_v2

random.seed(42)

backbone, cfg = get_base_model("resnet18")

# for dataset in ["factin", "footprocess", "lioness", "synaptic-semantic-segmentation"]:
for dataset in ["synaptic-semantic-segmentation"]:    

    train_dataset, _, testing_dataset = get_dataset(dataset, cfg, test_only=True)

    means, stds = [], []
    for img, _ in train_dataset:
        means.append(img.mean())
        stds.append(img.std())

    print(f"Dataset: {dataset}")
    print(f"Train mean: {numpy.mean(means)}")
    print(f"Train std: {numpy.mean(stds)}")

    # length = len(testing_dataset)
    # choices = random.sample(range(length), 10)
    # fig, axes = pyplot.subplots(2, 10, figsize=(20, 4))
    # for ax in axes.flatten():
    #     ax.axis("off")
    # for i, choice in enumerate(choices):
    #     image, mask = testing_dataset[choice]

    #     image = image.numpy()
    #     mask = mask.numpy()

    #     image = image.squeeze()
    #     if dataset == "factin":
    #         mask = mask.squeeze()[:-1]
    #     else:
    #         mask = mask.squeeze()

    #     composite = make_composite(mask, ["green", "magenta", "yellow", "cyan"][:len(mask)])
        
    #     axes[0, i].imshow(image, cmap="gray")
    #     axes[1, i].imshow(composite, cmap="hot")
    # fig.savefig(f"example-{dataset}.png", bbox_inches="tight", transparent=True, dpi=300)
    # pyplot.show()
        

