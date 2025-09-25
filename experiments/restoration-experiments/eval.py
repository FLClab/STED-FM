
import torch
import numpy
import random
import os

from collections import defaultdict
from tqdm import tqdm
from skimage.metrics import structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

def compute_scores(gt_images, pred_images, dataset_name: str):
    """
    Compute evaluation scores between ground truth and predicted images.

    :param gt_images: Ground truth images (tensor).
    :param pred_images: Predicted images (tensor).
    :param dataset_name: Name of the dataset being evaluated.

    :return: Dictionary of computed scores.
    """
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
    ssim = ms_ssim(pred_images, gt_images).cpu().numpy().squeeze().tolist()

    mse = torch.mean((gt_images - pred_images) ** 2, dim=[1,2,3])
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-9))
    psnr = psnr.cpu().numpy().squeeze().tolist()
    mse = mse.cpu().numpy().squeeze().tolist()
    mae = torch.mean(torch.abs(gt_images - pred_images), dim=[1,2,3]).cpu().numpy().squeeze().tolist()

    return {
        'msssim': ssim,
        'psnr': psnr,
        'mse': mse,
        'mae': mae
    }

def evaluate_denoising(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, savefolder: str, dataset_name: str):
    """
    Evaluate the denoising performance of the model on the given dataset.

    :param model: The denoising model to evaluate.
    :param dataloader: DataLoader providing (noisy_image, clean_image) pairs.
    :param device: The device to run the evaluation on (CPU or GPU).
    :param savefolder: Folder to save any outputs if needed.
    :param dataset_name: Name of the dataset being evaluated.

    :return: Dictionary containing evaluation metrics (e.g., average loss).
    """

    all_scores = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for i, (noisy_imgs, clean_imgs) in enumerate(tqdm(dataloader, desc="Evaluating Denoising")):

            if noisy_imgs.dim() == 3:
                noisy_imgs = noisy_imgs.unsqueeze(0)
                clean_imgs = clean_imgs.unsqueeze(0)

            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            outputs = model(noisy_imgs)

            scores = compute_scores(clean_imgs, outputs, dataset_name=dataset_name)
            for key, values in scores.items():
                all_scores[key].extend(values)

    return all_scores

if __name__ == "__main__":

    import argparse
    from datasets import get_dataset
    from stedfm.configuration import Configuration
    from stedfm.utils import update_cfg
    from stedfm import get_decoder, get_pretrained_model_v2
    from stedfm.DEFAULTS import BASE_PATH

    parser = argparse.ArgumentParser(description="Evaluate Denoising Model")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default="",
                    help="Model from which to restore from") 
    parser.add_argument("--dataset", required=True, type=str,
                    help="Name of the dataset to use")             
    parser.add_argument("--backbone", type=str, default=None,
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Backbone model to load")    
    parser.add_argument("--opts", nargs="+", default=[], 
                        help="Additional configuration options")
    parser.add_argument("--save-folder", type=str, default=f"{BASE_PATH}/denoising-baselines")
    
    args = parser.parse_args()
    # Assert args.opts is a multiple of 2
    if len(args.opts) == 1:
        args.opts = args.opts[0].split(" ")
    assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"
        # Ensure backbone weights are provided if necessary
    if args.backbone_weights in (None, "null", "None", "none"):
        args.backbone_weights = None
    if args.restore_from and not args.restore_from.endswith("/"):
        args.restore_from += "/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Try loading cfg file from restore-from arugments
    config_file = os.path.join(os.path.dirname(args.restore_from), "config.json")
    if os.path.isfile(config_file):
        cfg = Configuration.from_json(config_file)
        args.backbone = cfg["args"]["backbone"]
        args.backbone_weights = cfg["args"]["backbone_weights"]
    else:
        cfg = None
        assert args.backbone is not None, "Backbone must be provided"

    backbone, tmp_cfg = get_pretrained_model_v2(
        name=args.backbone, 
        weights=args.backbone_weights,
    )
    if cfg is None:
        cfg = tmp_cfg
    update_cfg(cfg, args.opts)

    # Loads dataset and dataset-specific configuration
    _, _, testing_dataset = get_dataset(
        name=args.dataset, 
        cfg=cfg
    )
    # Loads checkpoint
    checkpoint = torch.load(os.path.join(args.restore_from, "result.pt"))
    print(cfg)

    # Build the UNet model.
    model = get_decoder(backbone, cfg)
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("Restoring model...")
        model.load_state_dict(ckpt)
    else:
        raise ValueError
    model = model.to(device)

    # Build a PyTorch dataloader.
    test_loader = torch.utils.data.DataLoader(
        testing_dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=0,
        drop_last=False,
    )
    
    # Evaluate
    scores = evaluate_denoising(model, test_loader, device, args.save_folder, args.dataset)

    # Print average scores
    for metric, values in scores.items():
        mean = numpy.mean(values)
        std = numpy.std(values)
        print(f"{metric}: {mean:.4f} ± {std:.4f}")