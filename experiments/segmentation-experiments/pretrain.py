
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

from typing import Tuple, Any
from dataclasses import dataclass
from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from tqdm import tqdm
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
from torch.utils.data import Sampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from lightly.utils.scheduler import CosineWarmupScheduler

from decoders import get_decoder

from PiCIE.utils import *
from PiCIE.commons import *

import sys 
sys.path.insert(0, "../")

from datasets import get_dataset
from model_builder import get_base_model, get_pretrained_model_v2
from utils import update_cfg, save_cfg
from configuration import Configuration
from modules.transforms import PiCIETransform
from modules.transforms import batch_apply_transforms, apply_forward_transforms, apply_inverse_transforms

from DEFAULTS import BASE_PATH

def intensity_scale_(images: torch.Tensor, m: float = None, M: float = None) -> numpy.ndarray:
    """
    Helper function to scale the intensity of the images

    :param images: A `torch.Tensor` of the images to scale

    :returns : A `numpy.ndarray` of the scaled images
    """
    images = images.cpu().data.numpy()
    if m is None and M is None:
        return numpy.array([
            (image - image.min()) / (image.max() - image.min()) for image in images
        ])
    return (images - m) / (M - m)

class DatasetConfiguration(Configuration):

    num_classes: int = 1
    criterion: str = "MSELoss"
    min_annotated_ratio: float = 0.1

class SegmentationConfiguration(Configuration):
    
    freeze_backbone: bool = True
    num_epochs: int = 2
    learning_rate: float = 1e-3

class PiCIEConfiguration(Configuration):    

    K_train: int = 16
    max_iter: int = 30
    distance_metric: str = 'cosine'
    num_init_batches: int = 16
    num_batches: int = 16
    no_balance: bool = False
    mse: bool = False
    prop: bool = False
    lambda_prop: float = 10.0

# Define the configuration for the PiCIE model.
class PiCIETransformConfig(Configuration):

    input_size : int = 224
    cj_prob : float = 0.8
    cj_strength : float = 1.0
    cj_bright : float = 0.8
    cj_contrast : float = 0
    cj_sat : float = 0
    cj_hue : float = 0
    cj_gamma : float = 0
    scale : Tuple[float, float] = (1.0, 1.0)
    random_gray_scale : float = 0
    gaussian_blur : float = 0
    kernel_size : float = None
    sigmas : Tuple[float, float] = (0.1, 2)
    vf_prob : float = 0.5
    hf_prob : float = 0.5
    rr_prob : float = 0.5
    rr_degrees : float = None
    normalize : bool = False
    gaussian_noise_prob : float = 0.5
    gaussian_noise_mu: float = 0.
    gaussian_noise_std: float = 0.05
    poisson_noise_prob : float = 0.5
    poisson_noise_lambda : float = 0.5  


def train(cfg, writer, logger, dataloader, model, classifier1, classifier2, criterion1, criterion2, centroids1, centroids2, optimizer, epoch, device=torch.device("cpu")):
    losses = AverageMeter()
    losses_mse = AverageMeter()
    losses_cet = AverageMeter()
    losses_cet_across = AverageMeter()
    losses_cet_within = AverageMeter()
    losses_prop = AverageMeter()

    # switch to train mode
    model.train()
    if cfg.PiCIE.mse:
        criterion_mse = torch.nn.MSELoss().to(device)

    kl_loss = torch.nn.KLDivLoss(reduction='batchmean').to(device)

    metric_function1 = get_metric_as_conv(centroids1)
    metric_function2 = get_metric_as_conv(centroids2)

    classifier1.eval()
    classifier2.eval()
    for i_batch, views in enumerate(dataloader):

        (view1, bbox1, transforms1), (view2, bbox2, transforms2) = views
        view1 = view1.to(device)
        featmap1 = model.forward_features(view1)
        view2 = view2.to(device)
        featmap2 = model.forward_features(view2)

        # Inverses the geometric transformations
        view1 = batch_apply_transforms(apply_inverse_transforms, view1, transforms1)
        view2 = batch_apply_transforms(apply_inverse_transforms, view2, transforms2)
        featmap1 = batch_apply_transforms(apply_inverse_transforms, featmap1, transforms1)
        featmap2 = batch_apply_transforms(apply_inverse_transforms, featmap2, transforms2)

        if cfg.PiCIE.distance_metric == 'cosine':
            featmap1 = F.normalize(featmap1, dim=1, p=2)
            featmap2 = F.normalize(featmap2, dim=1, p=2)

        # Compute the labels
        # Note. In original code, the labels are saved to the disk and then reloaded here.
        # In this case, we are computing the labels again which will take slightly longer.
        B, C, _ = featmap1.size()[:3]

        label1 = predict_label(featmap1, centroids1, metric_function1).view(B, view1.size(2), view1.size(3))
        label1 = torch.LongTensor(label1).to(device)
        label2 = predict_label(featmap2, centroids2, metric_function2).view(B, view2.size(2), view2.size(3))
        label2 = torch.LongTensor(label2).to(device)

        # We herein create a mask from each bounding boxes in the two views
        # We need one mask per views
        def make_mask_from_bbox(mask, bboxA, bboxB):
            yA, xA, hA, wA = bboxA
            yB, xB, hB, wB = bboxB
            if xA < xB:
                min_col = xB - xA
                max_col = min(xA + wA, xB + wB) - xA
            else:
                min_col = 0
                max_col = min(xA + wA, xB + wB) - xA
            if yA < yB:
                min_row = yB - yA
                max_row = min(yA + hA, yB + hB) - yA
            else:
                min_row = 0
                max_row = min(yA + hA, yB + hB) - yA
            min_row, max_row, min_col, max_col = int(min_row), int(max_row), int(min_col), int(max_col)
            mask[:, min_row:max_row, min_col:max_col] = True
            return mask

        masks1, masks2 = [], []
        for i in range(len(bbox1)):
            mask1 = torch.zeros((1, view1.size(2), view1.size(3)), dtype=torch.bool)
            mask1 = make_mask_from_bbox(mask1, bbox1[i], bbox2[i])
            masks1.append(mask1)

            mask2 = torch.zeros((1, view2.size(2), view2.size(3)), dtype=torch.bool)
            mask2 = make_mask_from_bbox(mask2, bbox2[i], bbox1[i])
            masks2.append(mask2)

        masks1 = torch.cat(masks1, dim=0).to(device)
        masks2 = torch.cat(masks2, dim=0).to(device)

        if i_batch == 0:
            logger.info('Batch input size   : {}'.format(list(view1.shape)))
            logger.info('Batch label size   : {}'.format(list(label1.shape)))
            logger.info('Batch feature size : {}\n'.format(list(featmap1.shape)))

            if args.use_tensorboard:
                writer.add_images("views/views1", intensity_scale_(view1[:8]), epoch, dataformats="NCHW")
                writer.add_images("views/views2", intensity_scale_(view2[:8]), epoch, dataformats="NCHW")

                # writer.add_images("views/featmap1", intensity_scale_(view1[:8]), epoch, dataformats="NCHW")
                # writer.add_images("views/featmap2", intensity_scale_(view2[:8]), epoch, dataformats="NCHW")

                writer.add_images("labels/label1", intensity_scale_(label1[:8].unsqueeze(1), m=0, M=cfg.PiCIE.K_train), epoch, dataformats="NCHW")
                writer.add_images("labels/label2", intensity_scale_(label2[:8].unsqueeze(1), m=0, M=cfg.PiCIE.K_train), epoch, dataformats="NCHW")

                writer.add_images("masks/mask1", masks1[:8].unsqueeze(1), epoch, dataformats="NCHW")
                writer.add_images("masks/mask2", masks2[:8].unsqueeze(1), epoch, dataformats="NCHW")

        masks1 = feature_flatten(masks1).flatten()
        masks2 = feature_flatten(masks2).flatten()

        featmap12_processed, label12_processed = featmap1, label2.flatten()
        featmap21_processed, label21_processed = featmap2, label1.flatten()

        # Cross-view loss
        output12 = feature_flatten(classifier2(featmap12_processed)) # NOTE: classifier2 is coupled with label2
        output21 = feature_flatten(classifier1(featmap21_processed)) # NOTE: classifier1 is coupled with label1
        
        loss12  = criterion2(output12[masks1], label12_processed[masks2])
        loss21  = criterion1(output21[masks2], label21_processed[masks1])  

        loss_across = (loss12 + loss21) / 2.
        losses_cet_across.update(loss_across.item(), B)

        featmap11_processed, label11_processed = featmap1, label1.flatten()
        featmap22_processed, label22_processed = featmap2, label2.flatten()
        
        # Within-view loss
        output11 = feature_flatten(classifier1(featmap11_processed)) # NOTE: classifier1 is coupled with label1
        output22 = feature_flatten(classifier2(featmap22_processed)) # NOTE: classifier2 is coupled with label2

        loss11 = criterion1(output11[masks1], label11_processed[masks1])
        loss22 = criterion2(output22[masks2], label22_processed[masks2])

        loss_within = (loss11 + loss22) / 2. 
        losses_cet_within.update(loss_within.item(), B)
        loss = (loss_across + loss_within) / 2.
        
        losses_cet.update(loss.item(), B)
        
        if cfg.PiCIE.mse:
            loss_mse = criterion_mse(feature_flatten(featmap1)[masks1], feature_flatten(featmap2)[masks2])
            losses_mse.update(loss_mse.item(), B)

            loss = (loss + loss_mse) / 2. 

        if cfg.PiCIE.prop:
            label1 = torch.nn.functional.log_softmax(feature_flatten(predict_label_with_gradient(featmap1, centroids1, metric_function1).to(device)), dim=1)
            label2 = torch.nn.functional.log_softmax(feature_flatten(predict_label_with_gradient(featmap2, centroids2, metric_function2).to(device)), dim=1)

            # loss on the number of pixels from each class
            target = 1 / cfg.PiCIE.K_train * torch.ones(len(label1), cfg.PiCIE.K_train).to(device)
            loss_prop = (kl_loss(label1[masks1], target[masks1]) + kl_loss(label2[masks2], target[masks2])) / 2
            losses_prop.update(loss_prop.item(), B)

            # loss_prop = (torch.nn.functional.mse_loss(bincount1, target) + torch.nn.functional.mse_loss(bincount2, target)) / 2
            loss = loss + cfg.PiCIE.lambda_prop * loss_prop

        # record loss
        losses.update(loss.item(), B)

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i_batch % 200) == 0:
            logger.info('{0} / {1}\t'.format(i_batch, len(dataloader)))

    return losses.avg, losses_cet.avg, losses_cet_within.avg, losses_cet_across.avg, losses_mse.avg, losses_prop.avg

def main(args):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Loads backbone model
    if args.backbone_weights:
        backbone, cfg = get_pretrained_model_v2(args.backbone, weights=args.backbone_weights)
    else:
        backbone, cfg = get_base_model(args.backbone)

    # Updates configuration with additional options; performs inplace
    cfg.args = args
    segmentation_cfg = SegmentationConfiguration()
    for key, value in segmentation_cfg.__dict__.items():
        setattr(cfg, key, value)
    cfg.backbone_weights = args.backbone_weights
    cfg.dataset_cfg = DatasetConfiguration()
    cfg.transform = PiCIETransformConfig()
    cfg.PiCIE = PiCIEConfiguration()
    print(f"Config: {cfg.__dict__}")
    update_cfg(cfg, args.opts)

    model_name = "PiCIE"
    OUTPUT_FOLDER = os.path.join(args.save_folder, args.backbone, args.dataset, model_name)
    if args.dry_run and not args.restore_from:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))
    # Save and print configuration
    cfg.save(os.path.join(OUTPUT_FOLDER, "config.json"))
    print(cfg)

    logger = set_logger(os.path.join(OUTPUT_FOLDER, "train.log"))

    # Build the UNet model.
    model = get_decoder(backbone, cfg)
    model = model.to(DEVICE)

    transform = PiCIETransform(**cfg.transform.to_dict())
    training_dataset = get_dataset(args.dataset, args.dataset_path, debug=args.dry_run, transform=transform, mode='train')

    # Build a PyTorch dataloader.
    trainloader = torch.utils.data.DataLoader(
        training_dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4,
        drop_last=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.learning_rate, weight_decay=1e-2)
    logger.info(args)

    # Use random seed.
    fix_seed_for_reproducability(args.seed)

    # Start time.
    t_start = time.time()

    # Train start.
    stats = defaultdict(list)
    for epoch in range(0, cfg.num_epochs):

        logger.info('\n============================= [Epoch {}] =============================\n'.format(epoch))
        logger.info('Start computing centroids.')
        t1 = time.time()
        centroids1, kmloss1 = run_mini_batch_kmeans(cfg, logger, trainloader, model, view=1, device=DEVICE)
        centroids2, kmloss2 = run_mini_batch_kmeans(cfg, logger, trainloader, model, view=2, device=DEVICE)
        logger.info('-Centroids ready. [Loss: {:.5f}| {:.5f}/ Time: {}]\n'.format(kmloss1, kmloss2, get_datetime(int(time.time())-int(t1))))
        
        # Compute cluster assignment. 
        t2 = time.time()
        weight1 = compute_labels(cfg, logger, trainloader, model, centroids1, view=1, device=DEVICE)
        weight2 = compute_labels(cfg, logger, trainloader, model, centroids2, view=2, device=DEVICE)
        logger.info('-Weights: [{}] | [{}]'.format(weight1, weight2))
        logger.info('-Cluster labels ready. [{}]\n'.format(get_datetime(int(time.time())-int(t2)))) 
        
        # Criterion.
        if not cfg.PiCIE.no_balance:
            criterion1 = torch.nn.CrossEntropyLoss(weight=weight1).to(DEVICE)
            criterion2 = torch.nn.CrossEntropyLoss(weight=weight2).to(DEVICE)
        else:
            criterion1 = torch.nn.CrossEntropyLoss().to(DEVICE)
            criterion2 = torch.nn.CrossEntropyLoss().to(DEVICE)

        # Setup nonparametric classifier.
        classifier1 = initialize_classifier(cfg, centroids1.size(1))
        classifier2 = initialize_classifier(cfg, centroids2.size(1))
        classifier1.module.weight.data = centroids1.unsqueeze(-1).unsqueeze(-1)
        classifier2.module.weight.data = centroids2.unsqueeze(-1).unsqueeze(-1)
        freeze_all(classifier1)
        freeze_all(classifier2)

        # Delete since no longer needed. 
        # del centroids1 
        # del centroids2

        logger.info('Start training ...')
        train_loss, train_cet, cet_within, cet_across, train_mse, train_prop = train(cfg, writer, logger, trainloader, model, classifier1, classifier2, criterion1, criterion2, centroids1, centroids2, optimizer, epoch, device=DEVICE) 
        acc1, res1 = 0, {"mean_iou": 0}
        acc2, res2 = 0, {"mean_iou": 0}
        # acc1, res1 = evaluate(args, logger, testloader, classifier1, model)
        # acc2, res2 = evaluate(args, logger, testloader, classifier2, model)
        
        logger.info('============== Epoch [{}] =============='.format(epoch))
        logger.info('  Time: [{}]'.format(get_datetime(int(time.time())-int(t1))))
        logger.info('  K-Means loss   : {:.5f} | {:.5f}'.format(kmloss1, kmloss2))
        logger.info('  Training Total Loss  : {:.5f}'.format(train_loss))
        logger.info('  Training CE Loss (Total | Within | Across) : {:.5f} | {:.5f} | {:.5f}'.format(train_cet, cet_within, cet_across))
        logger.info('  Training MSE Loss (Total) : {:.5f}'.format(train_mse))
        logger.info('  Training Proportion Loss (Total) : {:.5f}'.format(train_prop))
        logger.info('  [View 1] ACC: {:.4f} | mIoU: {:.4f}'.format(acc1, res1['mean_iou']))
        logger.info('  [View 2] ACC: {:.4f} | mIoU: {:.4f}'.format(acc2, res2['mean_iou']))
        logger.info('========================================\n')
        
        stats["train_loss"].append(train_loss)
        stats["train_cet"].append(train_cet)
        stats["cet_within"].append(cet_within)
        stats["cet_across"].append(cet_across)
        stats["train_mse"].append(train_mse)
        stats["train_prop"].append(train_prop)
        stats["acc1"].append(acc1)
        stats["res1"].append(res1)
        stats["acc2"].append(acc2)
        stats["res2"].append(res2)

        if args.use_tensorboard:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/cet", train_cet, epoch)
            writer.add_scalar("train/cet_within", cet_within, epoch)
            writer.add_scalar("train/cet_across", cet_across, epoch)
            writer.add_scalar("train/mse", train_mse, epoch)
            writer.add_scalar("train/prop", train_prop, epoch)
            writer.add_scalar("train/acc1", acc1, epoch)
            writer.add_scalar("train/acc2", acc2, epoch)
            writer.add_scalar("train/mIoU1", res1['mean_iou'], epoch)
            writer.add_scalar("train/mIoU2", res2['mean_iou'], epoch)

        savedata = {
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "stats" : stats,
        }
        torch.save(
            savedata, 
            os.path.join(OUTPUT_FOLDER, f"result.pt"))
        del savedata

        # torch.save({'epoch': epoch+1, 
        #             'args' : args,
        #             'state_dict': model.state_dict(),
        #             'classifier1_state_dict' : classifier1.state_dict(),
        #             'classifier2_state_dict' : classifier2.state_dict(),
        #             'optimizer' : optimizer.state_dict(),
        #             },
        #             os.path.join(args.save_model_path, 'checkpoint_{}.pth.tar'.format(epoch)))
        
        # torch.save({'epoch': epoch+1, 
        #             'args' : args,
        #             'state_dict': model.state_dict(),
        #             'classifier1_state_dict' : classifier1.state_dict(),
        #             'classifier2_state_dict' : classifier2.state_dict(),
        #             'optimizer' : optimizer.state_dict(),
        #             },
        #             os.path.join(args.save_model_path, 'checkpoint.pth.tar'))
        
        # # Evaluate.
        # trainset    = get_dataset(args, mode='eval_val')
        # trainloader = torch.utils.data.DataLoader(trainset, 
        #                                             batch_size=args.batch_size_cluster,
        #                                             shuffle=True,
        #                                             num_workers=args.num_workers,
        #                                             pin_memory=True,
        #                                             collate_fn=collate_train,
        #                                             worker_init_fn=worker_init_fn(args.seed))

        # testset    = get_dataset(args, mode='eval_test')
        # testloader = torch.utils.data.DataLoader(testset, 
        #                                         batch_size=args.batch_size_test,
        #                                         shuffle=False,
        #                                         num_workers=args.num_workers,
        #                                         pin_memory=True,
        #                                         collate_fn=collate_eval,
        #                                         worker_init_fn=worker_init_fn(args.seed))


        # # Evaluate with fresh clusters.
        # acc_list_new = []  
        # res_list_new = []                 
        # logger.info('Start computing centroids.')
        # if args.repeats > 0:
        #     for _ in range(args.repeats):
        #         t1 = time.time()
        #         centroids1, kmloss1 = run_mini_batch_kmeans(args, logger, trainloader, model, view=-1)
        #         logger.info('-Centroids ready. [Loss: {:.5f}/ Time: {}]\n'.format(kmloss1, get_datetime(int(time.time())-int(t1))))
                
        #         classifier1 = initialize_classifier(args)
        #         classifier1.module.weight.data = centroids1.unsqueeze(-1).unsqueeze(-1)
        #         freeze_all(classifier1)
                
        #         acc_new, res_new = evaluate(args, logger, testloader, classifier1, model)
        #         acc_list_new.append(acc_new)
        #         res_list_new.append(res_new)
        # else:
        #     acc_new, res_new = evaluate(args, logger, testloader, classifier1, model)
        #     acc_list_new.append(acc_new)
        #     res_list_new.append(res_new)

        # logger.info('Average overall pixel accuracy [NEW] : {:.3f} +/- {:.3f}.'.format(np.mean(acc_list_new), np.std(acc_list_new)))
        # logger.info('Average mIoU [NEW] : {:.3f} +/- {:.3f}. '.format(np.mean([res['mean_iou'] for res in res_list_new]), 
        #                                                             np.std([res['mean_iou'] for res in res_list_new])))
        logger.info('Experiment done. [{}]\n'.format(get_datetime(int(time.time())-int(t_start))))
        


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default=None,
                    help="Model from which to restore from") 
    parser.add_argument("--save-folder", type=str, default=f"{BASE_PATH}/segmentation-baselines",
                    help="Model from which to restore from")     
    parser.add_argument("--dataset", type=str, default="STED",
                    help="Model from which to restore from")         
    parser.add_argument("--dataset-path", type=str, default="./data/FLCDataset/20240718-dataset-full-images.tar",
                    help="Model from which to restore from")             
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Backbone model to load")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Logging using tensorboard")
    parser.add_argument("--label-percentage", type=float, default=None,
                        help="Percentage of labels to use")
    parser.add_argument("--num-per-class", type=int, default=None,
                        help="Number of samples to use")
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

    main(args)
    