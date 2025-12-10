
import tifffile
import numpy
import torch
import random
import os
import glob
import pickle
import copy
from typing import Union

from tqdm.auto import trange, tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from read_roi import read_roi_file
from torch.utils.data import DataLoader
from skimage import measure
from stedfm import get_pretrained_model_v2
from stedfm.DEFAULTS import BASE_PATH
from classification import ImageDataset, OnTheFlySampler, PredictionBuilder
from matplotlib import pyplot

MAX_ANNOTATED_PATCHES_PER_IMAGE = 500
MAX_F1_SCORE = 0.9

class BestModelCheckpointer:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.best_f1 = 0.0

        self.history_of_models = []

    def update(self, clf: RandomForestClassifier, X_train: numpy.ndarray, y_train: numpy.ndarray) -> None:
        self.history_of_models.append({
            "clf" : copy.deepcopy(clf),
            "X_train": X_train.copy(),
            "y_train": y_train.copy()
        })
        self.save()

    def save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.history_of_models, f)

def make_label_from_roi(image_name: str, image_shape: tuple[int, int], patch_size: int=14) -> numpy.ndarray:
    
    label = numpy.zeros(image_shape, dtype=numpy.uint8)[numpy.newaxis]
    roi_name = image_name.replace(".tif", ".roi")

    dirname = os.path.dirname(roi_name)
    basename = os.path.basename(roi_name)

    try:
        roi_name = os.path.join(dirname, "AB-annotations", basename.replace("overview", "annotations").replace(".roi", ".tif"))
        if not os.path.exists(roi_name):
            raise FileNotFoundError
        label = tifffile.imread(roi_name)
        label = (label > 0).astype(numpy.uint8)[numpy.newaxis]
        return label
    except FileNotFoundError:
        roi_name = os.path.join(dirname, "AB-updated", basename)

    if os.path.exists(roi_name):
        rois = read_roi_file(roi_name)
        for roi in rois.values():
            if roi["type"] == "point":
                y, x = roi["y"], roi["x"]
                label[0, y, x] = 1
                # Makes a 3x3 square around the point
                for y_, x_ in zip(y, x):
                    label[0, max(0, y_-patch_size//2):y_+patch_size//2+1, max(0, x_-patch_size//2):x_+patch_size//2+1] = 1
    return label

def get_embeddings(model, cfg, image_names: list[str], size:int=224, patch_size:int=14) -> dict:

    embeddings = {}
    for image_name in image_names:
        if isinstance(image_name, str):
            image = tifffile.imread(image_name)
            if image.ndim == 3:
                image = image[0]
            label = make_label_from_roi(image_name, image.shape, patch_size=patch_size)

        # if cfg.in_channels != 3:
        m, M = numpy.min(image, axis=(-2, -1), keepdims=True), numpy.max(image, axis=(-2, -1), keepdims=True)
        image = (image - m) / (M - m + 1e-6)
        image = numpy.clip(image, 0, 1)

        dataset = ImageDataset(image, label, in_channels=cfg.in_channels, size=size, step=int(size * 1.0))
        sampler = OnTheFlySampler(dataset)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
        builder = PredictionBuilder(image.shape, size, num_classes=1)
        for X, y, positions in loader:

            if X.ndim == 3:
                X = X.unsqueeze(1)
            X = X.to(next(model.parameters()).device)
            features = model.forward_features(X).unsqueeze(1)
            features = features.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            batch_size = features.shape[0]
            label_per_patch = measure.block_reduce(y[:, 0], (1, 16, 16), numpy.mean).flatten()
            label_per_patch = (label_per_patch > 0.25).astype(numpy.uint8)

            patches_coords = []
            num_patches = 14
            for j, i in zip(*positions):
                j, i = int(j) // 224, int(i) // 224
                for dy in range(num_patches):
                    for dx in range(num_patches):
                        patches_coords.append((j * num_patches + dy, i * num_patches + dx))
            patches_coords = numpy.array(patches_coords)
            features = numpy.reshape(features, (-1, features.shape[-1]))    
            embeddings[image_name] = {
                "features": features,
                "labels" : label_per_patch,
                "coords": patches_coords
            }
    return embeddings

def get_clf(image_step: int) -> Union[RandomForestClassifier, None]:
    # clf_path = f"outputs/best_rf_model-{image_step - 1}.pkl"
    # if os.path.isfile(clf_path):
    #     clf = pickle.load(open(clf_path, "rb"))
    #     _ = clf.set_params(n_estimators=clf.n_estimators + 50, warm_start=True)
    # else:
    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, max_depth=5)
    return clf

def add_negative_samples(negative_groups, X_train: numpy.ndarray, y_train: numpy.ndarray, 
                         positive_indices: list[int], negative_indices_subset: list[int],
                         best_clf, best_model_f1, prev_X_train=None, prev_y_train=None,
                         image_step: int=0) -> tuple[numpy.ndarray, numpy.ndarray]:
    
    tentative_best_f1 = 0.0
    tentative_best_group = None    
    skipped_groups = 0
    added_negative = False
    to_add_idx = None
    for group_id, group in tqdm(negative_groups.items(), desc="Sampling from groups", leave=False):

        weights = None
        if best_clf is not None:
            X_train_subset = X_train[group]
            y_train_subset = y_train[group]
            pred = best_clf.predict(X_train_subset)
            if numpy.sum(pred == 1) / len(group) < 0.01:
                # Skip groups that are mostly predicted as negative
                skipped_groups += 1
                continue
            # Compute weights based on current best classifier
            weights = best_clf.predict_proba(X_train_subset)[:, 1]
            weights[weights < numpy.quantile(weights, 0.5)] = 0.0  # Boost weights above average
            weights = weights / numpy.sum(weights)

        idx = random.choices(group, weights=weights, k=1)[0]
        if idx in negative_indices_subset:
            n_attempts = 5
            for _ in range(n_attempts):
                idx = random.choices(group, weights=weights, k=1)[0]
                if idx not in negative_indices_subset:
                    break
            if idx in negative_indices_subset:
                continue
        tentative_negative_indices_subset = negative_indices_subset + [idx]
        indices_subset = positive_indices + tentative_negative_indices_subset
        if len(indices_subset) == 0:
            break
        
        try:
            X_train_subset = X_train[indices_subset]
            y_train_subset = y_train[indices_subset]
        except Exception as e:
            print("Error indexing training data with indices subset:", indices_subset)
            raise e

        clf = get_clf(image_step)
        if prev_X_train is not None and prev_y_train is not None:
            X_train_subset = numpy.concatenate([X_train_subset, prev_X_train], axis=0)
            y_train_subset = numpy.concatenate([y_train_subset, prev_y_train], axis=0)
        clf.fit(X_train_subset, y_train_subset)

        # Calculate F1-score
        pred = clf.predict(X_train)
        f1 = f1_score(y_train, pred)
        if f1 > best_model_f1:
            added_negative = True
            best_clf = copy.deepcopy(clf)
            best_model_f1 = f1
            negative_indices_subset = tentative_negative_indices_subset
            print("\nNew best model found with F1-score: {:0.2f}".format(best_model_f1))
            print("\tUsing group {}".format(group_id))
            print("\tPrecision: {:0.2f}, Recall: {:0.2f}, F1-score: {:0.2f}".format(precision_score(y_train, pred), recall_score(y_train, pred), f1_score(y_train, pred)))
            print("\tUsing {} positive and {} negative samples ({} total)".format(
                len(positive_indices),
                len(negative_indices_subset),
                len(positive_indices) + len(negative_indices_subset)
            ))
            print("\tUsing {:0.2f}% of patches".format((len(positive_indices) + len(negative_indices_subset)) / len(y_train) * 100))
            pickle.dump(clf, open(f"outputs/best_rf_model-{image_step}.pkl", "wb"))
            model_checkpointer.update(best_clf, X_train_subset, y_train_subset)
        if f1 > tentative_best_f1:
            tentative_best_f1 = f1
            tentative_best_group = group_id
            to_add_idx = idx

    return negative_indices_subset, best_clf, best_model_f1, skipped_groups, added_negative, to_add_idx, tentative_best_f1, tentative_best_group


def add_positive_samples(positive_groups, X_train: numpy.ndarray, y_train: numpy.ndarray, 
                         positive_indices_subset: list[int], negative_indices: list[int],
                         best_clf, best_model_f1, prev_X_train=None, prev_y_train=None, image_step:int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
    
    tentative_best_f1 = 0.0
    tentative_best_group = None    
    skipped_groups = 0
    added_positive = False
    to_add_idx = None
    for group_id, group in tqdm(positive_groups.items(), desc="Sampling from groups", leave=False):

        weights = None
        if best_clf is not None:
            X_train_subset = X_train[group]
            y_train_subset = y_train[group]
            pred = best_clf.predict(X_train_subset)
            if numpy.sum(pred == 0) / len(group) < 0.01:
                # Skip groups that are mostly predicted as positive
                skipped_groups += 1
                continue
            # Compute weights based on current best classifier
            weights = best_clf.predict_proba(X_train_subset)[:, 0]
            weights[weights < numpy.quantile(weights, 0.5)] = 0.0  # Boost weights above average
            weights = weights / numpy.sum(weights)

        idx = random.choices(group, weights=weights, k=1)[0]
        if idx in positive_indices_subset:
            n_attempts = 5
            for _ in range(n_attempts):
                idx = random.choices(group, weights=weights, k=1)[0]
                if idx not in positive_indices_subset:
                    break
            if idx in positive_indices_subset:
                continue
        tentative_positive_indices_subset = positive_indices_subset + [idx]
        indices_subset = negative_indices + tentative_positive_indices_subset
        if len(indices_subset) == 0:
            break

        X_train_subset = X_train[indices_subset]
        y_train_subset = y_train[indices_subset]

        # clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, max_depth=10)
        clf = get_clf(image_step)
        if prev_X_train is not None and prev_y_train is not None:
            X_train_subset = numpy.concatenate([X_train_subset, prev_X_train], axis=0)
            y_train_subset = numpy.concatenate([y_train_subset, prev_y_train], axis=0)
        clf.fit(X_train_subset, y_train_subset)

        # Calculate F1-score
        pred = clf.predict(X_train)
        f1 = f1_score(y_train, pred)
        if f1 > best_model_f1:
            added_positive = True
            best_clf = copy.deepcopy(clf)
            best_model_f1 = f1
            positive_indices_subset = tentative_positive_indices_subset
            print("\nNew best model found with F1-score: {:0.2f}".format(best_model_f1))
            print("\tUsing group {}".format(group_id))
            print("\tPrecision: {:0.2f}, Recall: {:0.2f}, F1-score: {:0.2f}".format(precision_score(y_train, pred), recall_score(y_train, pred), f1_score(y_train, pred)))
            print("\tUsing {} positive and {} negative samples ({} total)".format(
                len(positive_indices_subset),
                len(negative_indices),
                len(positive_indices_subset) + len(negative_indices)
            ))
            print("\tUsing {:0.2f}% of patches".format((len(positive_indices_subset) + len(negative_indices)) / len(y_train) * 100))
            pickle.dump(clf, open(f"outputs/best_rf_model-{image_step}.pkl", "wb"))
            model_checkpointer.update(best_clf, X_train_subset, y_train_subset)
        if f1 > tentative_best_f1:
            tentative_best_f1 = f1
            tentative_best_group = group_id
            to_add_idx = idx

    return positive_indices_subset, best_clf, best_model_f1, skipped_groups, added_positive, to_add_idx, tentative_best_f1, tentative_best_group

def sample_per_image(keys: Union[str, list[str]], image_names: list[str], embeddings: dict, args, prev_X_train: numpy.ndarray=None, prev_y_train: numpy.ndarray=None, prev_coords: numpy.ndarray=None, image_step: int=0) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:

    if isinstance(keys, str):
        keys = [keys]

    X_train, y_train, coords = [], [], []
    for key in keys:
        X_train.append(embeddings[key]["features"])
        y_train.append(embeddings[key]["labels"])
        coords.append(embeddings[key]["coords"])
    X_train = numpy.concatenate(X_train, axis=0)
    y_train = numpy.concatenate(y_train, axis=0)
    coords = numpy.concatenate(coords, axis=0)

    X_test = numpy.concatenate(
        [embeddings[name]["features"] for name in image_names if name not in keys],
        axis=0
    )
    y_test = numpy.concatenate(
        [embeddings[name]["labels"] for name in image_names if name not in keys],
        axis=0
    )

    positive_indices = numpy.where(y_train == 1)[0]
    negative_indices = numpy.where(y_train == 0)[0]

    random.shuffle(positive_indices)
    random.shuffle(negative_indices)

    positive_indices_groups = list(KMeans(n_clusters=16, random_state=args.seed).fit_predict(X_train[positive_indices]))
    positive_groups = {}
    for idx, group in zip(positive_indices, positive_indices_groups):
        if group not in positive_groups:
            positive_groups[group] = []
        positive_groups[group].append(idx)

    negative_indices_groups = list(KMeans(n_clusters=64, random_state=args.seed).fit_predict(X_train[negative_indices]))
    negative_groups = {}
    for idx, group in zip(negative_indices, negative_indices_groups):
        if group not in negative_groups:
            negative_groups[group] = []
        negative_groups[group].append(idx)
    
    if prev_X_train is not None and prev_y_train is not None:
        negative_indices_subset = []
        positive_indices_subset = []
    else:
        positive_indices_subset = random.choices(positive_indices, k=max(1, int(0.10*len(positive_indices))))
        negative_indices_subset = random.choices(negative_indices, k=max(1, int(0.10*len(positive_indices))))

    negative_positive_ratio = 2
    best_clf = None
    best_model_f1 = 0.0
    for _ in trange(len(positive_indices) * negative_positive_ratio, desc="Random searches for best model"):
        
        print("Adding negative samples...")

        negative_indices_subset, best_clf, best_model_f1, skipped_groups, added_negative, to_add_idx, tentative_best_f1, tentative_best_group = add_negative_samples(
            negative_groups, X_train, y_train, positive_indices_subset, negative_indices_subset, best_clf, best_model_f1, prev_X_train, prev_y_train, image_step
        )
        if not added_negative and to_add_idx is not None:
            # If no group improved the model, add the best one anyway
            negative_indices_subset.append(to_add_idx)
            print("\nNo group improved the model, adding best group anyway (group id: {}, f1: {:0.3f})".format(tentative_best_group, tentative_best_f1))
        print("\nSkipped {}/{} groups".format(skipped_groups, len(negative_groups)))
        # if skipped_groups == len(negative_groups):
        #     print("All groups were skipped, ending search.")
        #     break
        if best_model_f1 >= MAX_F1_SCORE:
            print("F1-score reached {}, ending search.".format(MAX_F1_SCORE))
            break
        if len(positive_indices_subset) + len(negative_indices_subset) >= MAX_ANNOTATED_PATCHES_PER_IMAGE:
            print("Reached maximum number of annotated patches per image ({}), ending search.".format(
                MAX_ANNOTATED_PATCHES_PER_IMAGE))
            break
        
        # Only add positive samples if we have previous data
        # if prev_X_train is not None and prev_y_train is not None:
        print("Adding positive samples...")
        positive_indices_subset, best_clf, best_model_f1, skipped_groups, added_positive, to_add_idx, tentative_best_f1, tentative_best_group = add_positive_samples(
            positive_groups, X_train, y_train, positive_indices_subset, negative_indices_subset, best_clf, best_model_f1, prev_X_train, prev_y_train, image_step
        )
        if not added_positive and to_add_idx is not None:
            # If no group improved the model, add the best one anyway
            positive_indices_subset.append(to_add_idx)
            print("\nNo group improved the model, adding best group anyway (group id: {}, f1: {:0.3f})".format(tentative_best_group, tentative_best_f1))
        print("\nSkipped {}/{} groups".format(skipped_groups, len(positive_groups)))
        # if skipped_groups == len(positive_groups):
        #     print("All groups were skipped, ending search.")
        #     break
        if best_model_f1 >= MAX_F1_SCORE:
            print("F1-score reached {}, ending search.".format(MAX_F1_SCORE))
            break
        if len(positive_indices_subset) + len(negative_indices_subset) >= MAX_ANNOTATED_PATCHES_PER_IMAGE:
            print("Reached maximum number of annotated patches per image ({}), ending search.".format(
                MAX_ANNOTATED_PATCHES_PER_IMAGE))
            break

        print("Current best F1-score: {:0.3f} using {} positive and {} negative samples ({} total)".format(
            best_model_f1, len(positive_indices_subset), len(negative_indices_subset), len(positive_indices_subset) + len(negative_indices_subset)))

    indices_subset = positive_indices_subset + negative_indices_subset
    X_train = X_train[indices_subset]
    y_train = y_train[indices_subset]
    coords = coords[indices_subset]
    return X_train, y_train, indices_subset, coords

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--backbone", type=str, default="mae-lightning-small",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Backbone model to load")
    parser.add_argument("--patch-size", type=int, default=14,
                        help="Patch size for label creation")
    parser.add_argument("--overwrite-embeddings", action="store_true",
                        help="Overwrite existing embeddings")
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
    image_names = glob.glob(os.path.join(BASE_PATH, "detection-data", "synapse-confocal-rf", "*.tif"))
    if args.overwrite_embeddings or not os.path.exists("outputs/embeddings.pkl"):
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

        embeddings = get_embeddings(model, cfg, image_names, size=224, patch_size=args.patch_size)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
    else:
        with open("outputs/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
    
    keys = list(embeddings.keys())

    image_step = 4

    prev_X_train, prev_y_train, prev_coords = None, None, None
    # if image_step > 0:
    #     prev_X_train, prev_y_train, prev_coords = [], [], []
    #     for idx in range(image_step):
    #         if os.path.exists(f"outputs/sampled_data_{idx}.pkl"):
    #             with open(f"outputs/sampled_data_{idx}.pkl", "rb") as f:
    #                 previous_data = pickle.load(f)
    #             prev_X_train.append(previous_data["X_train"])
    #             prev_y_train.append(previous_data["y_train"])
    #             prev_coords.append(previous_data["coords"])

    #     prev_X_train = numpy.concatenate(prev_X_train, axis=0)
    #     prev_y_train = numpy.concatenate(prev_y_train, axis=0)
    #     prev_coords = numpy.concatenate(prev_coords, axis=0)

    model_checkpointer = BestModelCheckpointer(save_path=f"outputs/best_rf_model-history-{image_step}.pkl")

    X_train, y_train, indices, coords = sample_per_image(
        keys[image_step], image_names, embeddings, args, 
        image_step=image_step, prev_X_train=prev_X_train, prev_y_train=prev_y_train)

    with open(f"outputs/sampled_data_{image_step}.pkl", "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "indices": indices,
            "coords": coords,
        }, f)