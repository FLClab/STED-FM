
import os
import glob
import random
import pickle
import numpy
import torch
import tifffile
import matplotlib.ticker as mticker

from sklearn.ensemble import RandomForestClassifier
from classification import Query
from stedfm import get_pretrained_model_v2
from stedfm.DEFAULTS import BASE_PATH
from matplotlib import pyplot
from tiffwrapper import make_composite
from read_roi import read_roi_file
from skimage import measure
from collections import defaultdict

from stedfm.utils import savefig

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

def bounding_boxes_intersect(boxA, boxB):
    # box = (minr, minc, maxr, maxc)
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, yB - yA + 1) * max(0, xB - xA + 1)

    return interArea > 0

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--backbone", type=str, default="mae-lightning-small",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Backbone model to load")    
    parser.add_argument("--image-training-idx", type=int, default=0,
                        help="Training image index")
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

    image_names = glob.glob(os.path.join(BASE_PATH, "detection-data", "synapse-confocal-rf", "*.tif"))
    images = [tifffile.imread(image_name) for image_name in image_names]
    images = {
        "all": {
            "image" : images,
            "label": [make_label_from_roi(image_name, images[i].shape) for i, image_name in enumerate(image_names)],
        }
    }

    box_size = 4.48  # in microns
    box_size_in_pixels = int(box_size / 0.080)
    image_training_idx = args.image_training_idx

    clf = pickle.load(open("outputs/best_rf_model-0.pkl", "rb"))
    history_of_models = pickle.load(open(f"outputs/best_rf_model-history-{image_training_idx}.pkl", "rb"))
    print(len(history_of_models), "models in history")
    scores_per_model = []
    for model_idx, model_info in enumerate(history_of_models):
        print(f"Model {model_idx}: trained on {model_info['X_train'].shape[0]} samples")
        clf = model_info["clf"]

        query = Query(images, class_id=0)
        scores = defaultdict(list)
        for i, query_result in enumerate(query.query(model, clf, cfg, step_factor=0.25)):

            coords = None
            if os.path.isfile(f"outputs/sampled_data_{i}.pkl"):
                output = pickle.load(open(f"outputs/sampled_data_{i}.pkl", "rb"))
                if "coords" in output:
                    coords = output["coords"]
            
            if coords is not None:
                annotated_ratio = len(coords) * (16 * 16) / (query_result["image"].shape[0] * query_result["image"].shape[1])
                print(f"Image {i} annotated ratio: {annotated_ratio:.4f}")
            query_result["example"] = numpy.arange(query_result["image"].size).reshape(query_result["image"].shape)
            query_result["example"][query_result["example"] < annotated_ratio * query_result["image"].size] = 1
            query_result["example"][query_result["example"] >= annotated_ratio * query_result["image"].size] = 0

            query_result["prediction"] = (query_result["prediction"] > 0.25).astype(numpy.uint8)

            composite = make_composite(
                [
                    query_result["image"],
                    query_result["prediction"][0],
                    query_result["label"][0],
                    # query_result["example"],
                ],
                luts=["gray", "cyan", "magenta", "yellow"][:3], ranges=[(0, numpy.quantile(query_result["image"], 0.995)), (0, 2.0), (0, 2.0), (0, 2.0)][:3]
            )
            # composite = make_composite(
            #     [
            #         query_result["image"],
            #         # query_result["prediction"][0],
            #         query_result["label"][0],
            #         # query_result["example"],
            #     ],
            #     luts=["gray", "magenta"], ranges=[(0, numpy.quantile(query_result["image"], 0.995)), (0, 2.0)]
            # )        

            fig, ax = pyplot.subplots(1, 1, figsize=(10, 10))
            ax.imshow(composite)
            patch_size = 16
            # if coords is not None:
            #     ax.scatter(coords[:, 1] * patch_size + patch_size // 2, coords[:, 0] * patch_size + patch_size // 2, s=100, c="#ffcc00", marker="x")
            ax.axis("off")
            pyplot.tight_layout()
            os.makedirs("outputs/visualizations", exist_ok=True)
            pyplot.savefig("outputs/visualizations/prediction_{}.png".format(i), dpi=600)
            pyplot.close()

            true_positives = []
            label = measure.label(query_result["label"][0])
            rprops = measure.regionprops(label)
            for rprop in rprops:
                minr, minc, maxr, maxc = rprop.bbox
                center_r = (minr + maxr) // 2
                center_c = (minc + maxc) // 2
                true_positives.append(
                    numpy.any(query_result["prediction"][0, center_r - box_size_in_pixels//2 : center_r + box_size_in_pixels//2,
                                                            center_c - box_size_in_pixels//2 : center_c + box_size_in_pixels//2])
                )
            print("True positives:", sum(true_positives))

            false_positives = []
            label = measure.label(query_result["prediction"][0])
            rprops = measure.regionprops(label)
            bounding_boxes = []
            for rprop in rprops:
                minr, minc, maxr, maxc = rprop.bbox
                center_r = (minr + maxr) // 2
                center_c = (minc + maxc) // 2
                false_positives.append(
                    not numpy.any(query_result["label"][0, center_r - box_size_in_pixels//2 : center_r + box_size_in_pixels//2,
                                                           center_c - box_size_in_pixels//2 : center_c + box_size_in_pixels//2])
                )
                bounding_boxes.append(
                    (center_r - box_size_in_pixels//2, center_c - box_size_in_pixels//2,
                     center_r + box_size_in_pixels//2, center_c + box_size_in_pixels//2)
                )
            print("False positives:", sum(false_positives))

            false_negatives = []
            label = measure.label(query_result["label"][0])
            rprops = measure.regionprops(label)
            for rprop in rprops:
                false_negatives.append(
                    not any(bounding_boxes_intersect(bbox, rprop.bbox) for bbox in bounding_boxes)
                )
            print("False negatives:", sum(false_negatives))

            precision = sum(true_positives) / (sum(true_positives) + sum(false_positives) + 1e-8)
            recall = sum(true_positives) / (sum(true_positives) + sum(false_negatives) + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            scores["precision"].append(precision)
            scores["recall"].append(recall)
            scores["f1"].append(f1)

        scores_per_model.append(scores)

        fig, ax = pyplot.subplots(figsize=(3,3))
        xs = [model_info['X_train'].shape[0] * (16 * 16) / (query_result["image"].shape[0] * query_result["image"].shape[1]) for model_info in history_of_models[:model_idx+1]]
        for score in ["f1", "precision", "recall"]:
            mean_score = numpy.mean([s[score] for s in scores_per_model[:model_idx+1]], axis=1)
            std_score = numpy.std([s[score] for s in scores_per_model[:model_idx+1]], axis=1)
            ax.plot(xs, mean_score, label=f'Mean {score.capitalize()}', marker='o')
            ax.fill_between(xs, mean_score - std_score, mean_score + std_score, alpha=0.3)
        ax.set_xlabel("Number of Training Samples")
        ax.set_ylabel("Score")
        
        formatter = mticker.PercentFormatter(xmax=1.0, decimals=1) # decimals=0 for no decimal places
        ax.xaxis.set_major_formatter(formatter)
        ax.legend()
        pyplot.tight_layout()
        savefig(fig, f"outputs/visualizations/model_{image_training_idx}_scores", dpi=600)
        pyplot.close()
    
    with open(f"outputs/scores_per_model_rf-{image_training_idx}.pkl", "wb") as file:
        pickle.dump(scores_per_model, file)