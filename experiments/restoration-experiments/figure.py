
import os
import glob
import tifffile
import numpy
import torch
import json

from PIL import Image
from matplotlib import pyplot

from stedfm.DEFAULTS import COLORS, MARKERS, BASE_PATH
from stedfm.utils import savefig
from stedfm.stats import resampling_stats, plot_p_values
from eval import compute_scores

COLORS.care2d = "gray"
COLORS.n2v = "gray"
COLORS.pix2pix = "gray"
COLORS.unet_rcan = "gray"

YLIMS = {
    "psnr" : (0, 40),
    "msssim" : (0., 1.0),
    "mse" : (0, 0.05),
    "mae" : (0, 0.20)
}

def simple_beeswarm(y, nbins=None, maxwidth=0.8):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = numpy.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = numpy.zeros(len(y))
    ylo = numpy.min(y)
    yhi = numpy.max(y)
    dy = (yhi - ylo) / nbins
    ybins = numpy.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = numpy.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[numpy.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + numpy.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + numpy.arange(len(b))) * -dx
    
    x = x / numpy.max(numpy.abs(x)) * maxwidth / 2
    return x

def get_ground_truth_images(dataset_name):
    if dataset_name == "ov-lqhq-mt":
        path = os.path.join(BASE_PATH, "denoising-data", "ov-lqhq-mt-tif", "fixed_cell_microtubule_u2os_alphatubulin_star635p_registered", "test_data", "ground_truth_image_patches")
    elif dataset_name == "ov-lqhq-live-mito":
        path = os.path.join(BASE_PATH, "denoising-data", "ov-lqhq-live-mito", "live_cell_mitochondria_u2os_tom20_halotag7_dm_sir", "test_and_training_data_1", "ground_truth_images")
    elif dataset_name == "jmb-lqhq":
        path = os.path.join(BASE_PATH, "denoising-data", "jmb-lqhq", "exported", "test", "gt")
    elif dataset_name in ["kt-lqhq-gephyrin", "kt-lqhq", "kt-lqhq-vgat"]:
        path = os.path.join(BASE_PATH, "denoising-data", "kt-lqhq", dataset_name, "exported", "test", "gt")
    elif dataset_name in ["kt-sr-vgat"]:
        path = os.path.join(BASE_PATH, "super-resolution-data", "kt-super-resolution", dataset_name, "exported", "test", "gt")
    elif dataset_name == "unet-rcan-tub":
        path = os.path.join(BASE_PATH, "denoising-data", "unet-rcan-lqhq", "unet-rcan-tub", "exported", "test", "gt")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return sorted(glob.glob(os.path.join(path, "*.tif")))

def get_raw_images(dataset_name, gt_images):
    if dataset_name == "ov-lqhq-mt":
        raw_images = [
            gt_image.replace("ground_truth_image_patches", "low_intensity_image_patches") for gt_image in gt_images
        ]
    elif dataset_name == "ov-lqhq-live-mito":
        raw_images = [
            gt_image.replace("ground_truth_images", "low_intensity_images") for gt_image in gt_images
        ]
    elif dataset_name in ["jmb-lqhq", "unet-rcan-tub", "kt-lqhq-gephyrin", "kt-lqhq-vgat", "kt-lqhq", "kt-sr-vgat"]:
        raw_images = [
            gt_image.replace(os.path.join("exported", "test", "gt"), os.path.join("exported", "test", "raw")) for gt_image in gt_images
        ]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return raw_images

def get_predicted_images(method, dataset_name, gt_images):
    if method in ["CARE2D", "N2V"]:
        path = os.path.join(BASE_PATH, "denoising-baselines", f"{method}-{dataset_name}", "Quality Control", "Prediction")
        predicted_images = []
        for gt_image in gt_images:
            filename = os.path.basename(gt_image)
            predicted_images.append(os.path.join(path, filename))
    elif method == "pix2pix":
        path = os.path.join(BASE_PATH, "denoising-baselines", f"{method}-{dataset_name}", "Quality Control", "latest")
        predicted_images = []
        tmp = []
        for gt_image in gt_images:
            filename = os.path.basename(gt_image)
            predicted_images.append(os.path.join(path, filename.replace(".tif", "_fake_B.png")))
            tmp.append(os.path.join(path, filename.replace(".tif", "_real_B.png")))
        gt_images = tmp
    elif method == "UNet-RCAN":
        gt_images = [
            os.path.join(BASE_PATH, "denoising-baselines", f"{method}-{dataset_name}", "evaluation", "gt.tif")
        ]
        predicted_images = [
            os.path.join(BASE_PATH, "denoising-baselines", f"{method}-{dataset_name}", "evaluation", "pred.tif")
        ]
    elif method in ["STED", "SIM", "HPA", "JUMP", "IMAGENET1K_V1"]:
        path = os.path.join(BASE_PATH, "denoising-baselines", "mae-lightning-small", f"{dataset_name}", f"pretrained-frozen-MAE_SMALL_{method}-42", "predictions", dataset_name)
        gt_images = glob.glob(os.path.join(path, "*_cleaned_images.tif"))
        predicted_images = glob.glob(os.path.join(path, "*_denoised_predictions.tif"))
    return gt_images, predicted_images

def load_images(image_paths):
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist.")
            continue
        if path.endswith(".tif"):
            img = tifffile.imread(path)
        elif path.endswith(".png"):
            img = Image.open(path).convert("L")
            img = numpy.array(img) / 255.0
        else:
            raise ValueError(f"Unsupported file format: {path}")

        if img.ndim == 2:
            img = img[numpy.newaxis]
        elif img.ndim == 3:
            img = img[:, numpy.newaxis, :, :]
        images.append(img)

    if len(images) == 0:
        return None
    if len(images)>1:
        images = numpy.stack(images, axis=0)
    else:
        images = images[0]

    m, M = images.min(axis=(-2, -1), keepdims=True), images.max(axis=(-2, -1), keepdims=True)
    images = (images - m) / (M - m + 1e-8)
    return images

def plot_scores(all_scores, dataset_name, encoder=None):
    metrics = list(all_scores[list(all_scores.keys())[0]].keys())
    num_metrics = len(metrics)

    for i, metric in enumerate(metrics):
        fig, ax = pyplot.subplots(figsize=(3, 3))
        samples = []
        for j, (method, scores) in enumerate(all_scores.items()):
            values = scores[metric]
            width = 0.8
            bplot = ax.boxplot(values, positions=[j], widths=width, showfliers=False, 
                               patch_artist=True, 
                               medianprops=dict(color=COLORS[method], linewidth=1), 
                               boxprops=dict(color=COLORS[method], facecolor=COLORS[method], linewidth=1), 
                               whiskerprops=dict(color=COLORS[method], linewidth=1), 
                               capprops=dict(color=COLORS[method], linewidth=1))
            # for whisker in bplot['whiskers']:
            #     whisker.set_color(COLORS[method])
            for patch in bplot['boxes']:
                # patch.set_facecolor(COLORS[method])
                patch.set_alpha(0.5)
            # ax.scatter(simple_beeswarm(values, maxwidth=width) + j, values, color=COLORS[method], label=method, alpha=0.7)
            samples.append(values)
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(all_scores)))
        ax.set_ylim(YLIMS[metric])
        ax.set_xticklabels(list(all_scores.keys()), rotation=45, ha="right")
        # ax.legend()


        savefig(fig, os.path.join("results", f"{dataset_name}_{metric}_{encoder}"))
        pyplot.close()
        
        p_values, F_p_value = resampling_stats(samples, list(all_scores.keys()))
        fig, ax = plot_p_values(p_values)
        savefig(fig, os.path.join("results", f"{dataset_name}_{metric}_{encoder}_stats"))
        pyplot.close()

def main():
    
    import argparse
    parser = argparse.ArgumentParser(description="Plot restoration experiment results")
    parser.add_argument("--dataset", required=True, type=str,
                        help="Name of the dataset to use")
    parser.add_argument("--encoder", required=False, type=str, default="pretrained-frozen", choices=["pretrained", "pretrained-frozen"],
                        help="Restoration method to evaluate")
    args = parser.parse_args()

    methods = [
        "CARE2D",
        "N2V",
        "pix2pix",
        "UNet-RCAN",
        # "IMAGENET1K_V1",
        # "JUMP",
        # "HPA",
        # "SIM",
        "STED",
    ]
    all_scores = {}
    for method in methods:
        print(f"Evaluating {method} on {args.dataset}")
        if method in ["STED", "SIM", "HPA", "JUMP", "IMAGENET1K_V1"]:
            path = os.path.join(BASE_PATH, "denoising-baselines", "mae-lightning-small", f"{args.dataset}", f"{args.encoder}-MAE_SMALL_{method}-42", "denoising-scores.json")
            if not os.path.exists(path):
                print(f"Skipping {method} on {args.dataset} due to missing scores file")
                scores = {"psnr": [], "msssim": [], "mse": [], "mae": []}
                continue
            else:
                scores = json.load(open(os.path.join(BASE_PATH, "denoising-baselines", "mae-lightning-small", f"{args.dataset}", f"{args.encoder}-MAE_SMALL_{method}-42", "denoising-scores.json"), "r"))
        else:
            gt_images = get_ground_truth_images(args.dataset)
            gt_images, predicted_images = get_predicted_images(method, args.dataset, gt_images)

            gt_images = load_images(gt_images)
            predicted_images = load_images(predicted_images)
            if gt_images is None or predicted_images is None:
                print(f"Skipping {method} on {args.dataset} due to missing images")
                continue
            if len(gt_images) != len(predicted_images):
                print(f"Skipping {method} on {args.dataset} due to mismatched number of images")
                continue

            gt_images = torch.from_numpy(gt_images).float()
            predicted_images = torch.from_numpy(predicted_images).float()

            scores = compute_scores(gt_images, predicted_images, dataset_name=args.dataset, size=224)
        all_scores[method] = scores

    print("Scores:")
    for method, scores in all_scores.items():
        print(f"Method: {method}")
        for metric, values in scores.items():
            mean_value = numpy.median(values)
            print(f"  {metric}: {mean_value:.4f}")

    if all_scores:
        plot_scores(all_scores, args.dataset, encoder=args.encoder)

if __name__ == "__main__":
    main()