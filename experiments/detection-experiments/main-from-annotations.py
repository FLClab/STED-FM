
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
import itertools

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import sys
# sys.path.insert(0, "../segmentation-experiments")
# from datasets import get_dataset

sys.path.insert(0, "..")

from DEFAULTS import BASE_PATH
from datasets import get_dataset
from model_builder import get_base_model, get_pretrained_model_v2
from utils import update_cfg, save_cfg
from configuration import Configuration

from classification import Query, Template

def aggregate_from_templates(templates):
    aggregated = defaultdict(list)
    keys = set()
    for template in templates:
        keys.update(template.keys())
    keys = sorted(list(keys))
    for key in keys:
        for template in templates:
            aggregated[key].extend(template.get(key, []))
    return aggregated

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--dataset", required=True, type=str,
                    help="Name of the dataset to use")             
    parser.add_argument("--backbone", type=str, default="mae-lightning-small",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default="MAE_SMALL_STED",
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


    testing_dataset = get_dataset(name=args.dataset, path=None, transform=None)
    print(f"Testing dataset: {args.dataset} ({len(testing_dataset)} images)")

    template = Template(testing_dataset.images, class_id=None, mode="all")
    templates = template.get_template(model, cfg)

    templates = aggregate_from_templates(templates)

    X = numpy.concatenate([values for values in templates.values()], axis=0).squeeze()
    y = numpy.concatenate([numpy.ones(len(values)) * i for i, values in enumerate(templates.values())], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, shuffle=True)
    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, max_depth=15)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    image_name_mapper = defaultdict(list)
    query = Query(testing_dataset.images, class_id=None)
    for i, (query_result) in enumerate(query.query(model, clf, cfg)):
        image = query_result["image"]
        label = query_result["label"]
        prediction = query_result["prediction"]
        image_path = query_result["image-name"]

        # fig, ax = pyplot.subplots(1, 3)
        # ax[0].imshow(image, cmap="gray", vmin=0, vmax=0.7*image.max())
        # ax[1].imshow(label[template.class_id], cmap="gray", vmin=0, vmax=1)
        # ax[2].imshow(prediction, cmap="hot", vmin=0, vmax=1)
        # fig.savefig(f"query.png")

        # threshold = numpy.quantile(prediction, 0.95)
        # prediction[prediction < threshold] = threshold

        if isinstance(image_path, str):
            image_name = os.path.basename(image_path)
            savepath = os.path.join(BASE_PATH, "results", "segmentation-experiments", args.backbone_weights, "classification", query_result["condition"], image_name)
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

            # Save metadata
            image_name_mapper[query_result["condition"]].append({
                "image-name" : image_name,
                "savepath" : os.path.relpath(savepath, BASE_PATH),
                "original-path" : os.path.relpath(image_path, BASE_PATH),
            })
            with open(os.path.join(os.path.join(BASE_PATH, "results", "segmentation-experiments", args.backbone_weights, "classification"), "metadata.json"), "w") as f:
                json.dump(image_name_mapper, f, indent=4)
        else:
            savepath = f"image-{i}-{args.backbone_weights}.tif"

        stack = numpy.stack([image, *prediction], axis=0)
        tiffwrapper.imwrite(
            savepath, stack.astype(numpy.float32), 
            composite=True, luts=["gray", *["magenta", "green", "cyan"][:len(prediction)]])
