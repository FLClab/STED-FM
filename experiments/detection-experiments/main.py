
import matplotlib.colors
import torch
import torchvision
import numpy
import os
import typing
import random
import dataclasses
import time
import json
import argparse
import uuid
import skimage.measure
import matplotlib
from scipy.spatial import distance
import tifffile
import tiffwrapper

from dataclasses import dataclass
from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from tqdm import tqdm
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from matplotlib import pyplot

import sys
sys.path.insert(0, "../segmentation-experiments")
from datasets import get_dataset

sys.path.insert(0, "..")

from model_builder import get_base_model, get_pretrained_model_v2
from utils import update_cfg, save_cfg
from configuration import Configuration

from template import Template, Query, sample_topk

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--dataset", required=True, type=str,
                    help="Name of the dataset to use")             
    parser.add_argument("--backbone", type=str, default="mae-lightning-small",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Backbone model to load")    
    parser.add_argument("--opts", nargs="+", default=[], 
                        help="Additional configuration options")
    parser.add_argument("--dry-run", action="store_true",
                        help="Activates dryrun")        
    args = parser.parse_args()

    # Assert args.opts is a multiple of 2
    if len(args.opts) == 1:
        args.opts = args.opts[0].split(" ")
    assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"
    # Ensure backbone weights are provided if necessary
    if args.backbone_weights in (None, "null", "None", "none"):
        args.backbone_weights = None

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Loads backbone model
    model, cfg = get_pretrained_model_v2(
        args.backbone, 
        weights=args.backbone_weights,
        as_classifier=True,
        num_classes=1,
        blocks="all",
        global_pool="patch",
        mask_ratio=0,
    )
    model = model.to(DEVICE)
    print(cfg)

    cfg.args = args
    update_cfg(cfg, args.opts)

    CLASS = "perforated"
    training_dataset, validation_dataset, testing_dataset = get_dataset(name=args.dataset, cfg=cfg)
    print("Classes: ", training_dataset.classes)
    print("Required class: ", CLASS)

    template = Template(training_dataset.images, class_id=training_dataset.classes.index(CLASS), mode="choice")
    template_query = template.get_template(model, cfg)

    # In mode "choice", template_query is a dictionary and we can extract the image and mask templates
    if isinstance(template_query, dict):
        image_template = template_query["image-template"]
        mask_template = template_query["mask-template"]

        if image_template.ndim == 2:
            stack = numpy.stack([image_template, mask_template], axis=0)
        else:
            stack = numpy.stack([image_template, mask_template], axis=1)
        tifffile.imwrite(f"template-{CLASS}-{args.backbone_weights}.tif", stack.astype(numpy.float32), imagej=True, metadata={"mode": "composite"})

        template_query = template_query["template"]

    query = Query(testing_dataset.images, class_id=testing_dataset.classes.index(CLASS))
    for i, (image, label, prediction) in enumerate(query.query(template_query, model, cfg)):
        # fig, ax = pyplot.subplots(1, 3)
        # ax[0].imshow(image, cmap="gray", vmin=0, vmax=0.7*image.max())
        # ax[1].imshow(label[template.class_id], cmap="gray", vmin=0, vmax=1)
        # ax[2].imshow(prediction, cmap="hot", vmin=0, vmax=1)
        # fig.savefig(f"query.png")

        threshold = numpy.quantile(prediction, 0.95)
        prediction[prediction < threshold] = threshold

        stack = numpy.stack([image, label[template.class_id], prediction], axis=0)
        tiffwrapper.imwrite(
            f"image-{CLASS}-{i}-{args.backbone_weights}.tif", stack.astype(numpy.float32), 
            composite=True, luts=["gray", "green", "magenta"])

        crops, coords = sample_topk(image, prediction, k=25, shape=224)
        tifffile.imwrite(f"crop-{CLASS}-{i}-{args.backbone_weights}.tif", numpy.array(crops).astype(numpy.float32), imagej=True)
        # break

    # random.seed(None)
    # idx = random.randint(0, len(testing_dataset)-1)
    # img, mask = testing_dataset[idx]
    # model.eval()
    # with torch.no_grad():
    #     img = img.unsqueeze(0).to(DEVICE)
    #     mask = mask.numpy().squeeze()[template.class_id]
    #     m = skimage.measure.block_reduce(mask, (16, 16), numpy.mean)
    #     patch_idx = numpy.argmax(m.ravel())
    #     features = model.forward_features(img)

    #     features = features.cpu().squeeze().numpy()
    #     distances = distance.cdist(features, query[numpy.newaxis], "cosine")
    #     distances = distances.ravel()


    # # idx = 67
    # # img, mask = training_dataset[idx]

    # # model.eval()
    # # with torch.no_grad():
    # #     img, mask = training_dataset[idx]
    # #     img = img.unsqueeze(0).to(DEVICE)

    # #     mask = mask.numpy().squeeze()
    # #     m = skimage.measure.block_reduce(mask, (16, 16), numpy.mean)
    # #     patch_idx = numpy.argmax(m.ravel())
    # #     features = model.forward_features(img)
    # #     query = features[0, patch_idx, :]

    # #     # Get a random image from the testing dataset
    # #     random.seed(None)
    # #     idx = random.randint(0, len(testing_dataset)-1)
    # #     img, mask = testing_dataset[idx]
    # #     img = img.unsqueeze(0).to(DEVICE)
    # #     mask = mask.numpy().squeeze()
    # #     m = skimage.measure.block_reduce(mask, (16, 16), numpy.mean)
    # #     patch_idx = numpy.argmax(m.ravel())
    # #     features = model.forward_features(img)

    # #     distances = torch.nn.functional.cosine_similarity(features[0], query.unsqueeze(0), dim=1)
    # #     distances = distances.cpu().numpy()

    # img = img.squeeze().cpu().numpy()
    # # mask[mask == 0] = numpy.nan

    # fig, axes = pyplot.subplots(1, 2)
    # axes[0].imshow(img, cmap="gray", vmin=0, vmax=0.7*img.max())
    # # ax.imshow(mask, alpha=0.3, cmap="hot")

    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["black", "#ffcc00"])
    # axes[1].imshow(img, cmap="gray", vmin=0, vmax=0.7*img.max())
    # axes[1].imshow(distances.reshape(14, 14), alpha=0.5, cmap="coolwarm", vmin=0, vmax=1, 
    #           extent=[0, 224, 224, 0])
    # for ax in axes.ravel():
    #     # ax.grid(True)
    #     major_ticks = numpy.arange(0, 224, 16)
    #     ax.set_xticks(major_ticks)
    #     ax.set_yticks(major_ticks)
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])

    # cmap = pyplot.get_cmap("hot")
    # norm = matplotlib.colors.Normalize(vmin=distances.min(), vmax=1)

    # # for j in range(14):
    # #     for i in range(14):
    # #         x = i + j * 14
    # #         color = cmap(norm(distances[x].item()))
    # #         ax.text(i*16 + 8, j*16 + 8, f"{x:.0f}", color=color, 
    # #                 fontsize=8, ha="center", va="center", fontweight="bold")

    # fig.savefig("test.png")        