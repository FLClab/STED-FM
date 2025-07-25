import random
import os 
import logging
import pickle 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# import faiss 

# from data.coco_train_dataset import TrainCOCO
# from data.coco_eval_dataset import EvalCOCO 
# from data.cityscapes_train_dataset import TrainCityscapes
# from data.cityscapes_eval_dataset import EvalCityscapes

################################################################################
#                                  General-purpose                             #
################################################################################

def str_list(l):
    return '_'.join([str(x) for x in l]) 

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_datetime(time_delta):
    days_delta = time_delta // (24*3600)
    time_delta = time_delta % (24*3600)
    hour_delta = time_delta // 3600 
    time_delta = time_delta % 3600 
    mins_delta = time_delta // 60 
    time_delta = time_delta % 60 
    secs_delta = time_delta 

    return '{}:{}:{}:{}'.format(days_delta, hour_delta, mins_delta, secs_delta)



################################################################################
#                                Metric-related ops                            #
################################################################################

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class) # Exclude unlabelled data.
    hist = np.bincount(n_class * label_true[mask] + label_pred[mask],\
                       minlength=n_class ** 2).reshape(n_class, n_class)
    
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist


def get_result_metrics(histogram):
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp 

    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn) 
    opc = np.sum(tp) / np.sum(histogram)

    result = {"iou": iou,
              "mean_iou": np.nanmean(iou),
              "precision_per_class (per class accuracy)": prc,
              "mean_precision (class-avg accuracy)": np.nanmean(prc),
              "overall_precision (pixel accuracy)": opc}

    result = {k: 100*v for k, v in result.items()}

    return result

def compute_negative_euclidean(featmap, centroids, metric_function):
    centroids = centroids.unsqueeze(-1).unsqueeze(-1)
    return - (1 - 2*metric_function(featmap)\
                + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 


def get_metric_as_conv(centroids):
    N, C = centroids.size()

    centroids_weight = centroids.unsqueeze(-1).unsqueeze(-1)
    metric_function  = nn.Conv2d(C, N, 1, padding=0, stride=1, bias=False)
    metric_function.weight.data = centroids_weight
    metric_function = nn.DataParallel(metric_function)
    metric_function = metric_function.cuda()
    
    return metric_function

################################################################################
#                                General torch ops                             #
################################################################################

def freeze_all(model):
    for param in model.module.parameters():
        param.requires_grad = False 


def initialize_classifier(cfg, in_dim):
    classifier = get_linear(in_dim, cfg.PiCIE.K_train)
    classifier = nn.DataParallel(classifier)
    classifier = classifier.cuda()

    return classifier

def get_linear(indim, outdim):
    classifier = nn.Conv2d(indim, outdim, kernel_size=1, stride=1, padding=0, bias=True)
    classifier.weight.data.normal_(0, 0.01)
    classifier.bias.data.zero_()

    return classifier


def feature_flatten(feats):
    if len(feats.size()) == 2:
        # feature already flattened. 
        return feats
    
    feats = feats.view(feats.size(0), feats.size(1), -1).transpose(2, 1)\
            .contiguous().view(-1, feats.size(1))
    
    return feats 

################################################################################
#                                   Faiss related                              #
################################################################################

def get_faiss_module(args):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False 
    cfg.device     = 0 #NOTE: Single GPU only. 
    idx = faiss.GpuIndexFlatL2(res, args.in_dim, cfg)

    return idx

def get_init_centroids(args, K, featlist, index):
    clus = faiss.Clustering(args.in_dim, K)
    clus.seed  = np.random.randint(args.seed)
    clus.niter = args.kmeans_n_iter
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)

    return faiss.vector_float_to_array(clus.centroids).reshape(K, args.in_dim)

def module_update_centroids(index, centroids):
    index.reset()
    index.add(centroids)

    return index 

def fix_seed_for_reproducability(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic. 

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068 
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi 
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)

################################################################################
#                               Training Pipelines                             #
################################################################################

def predict_label_with_gradient(featmap, centroids, metric_function):
    scores = compute_negative_euclidean(featmap, centroids, metric_function)
    return scores
    # print(scores.requires_grad, scores.shape)

    # print(scores.topk(1, dim=1)[1], scores.topk(1, dim=1)[1].requires_grad)

    # labels = []
    # for idx in range(scores.size(0)):
    #     labels.append(scores[idx].topk(1, dim=0)[1].flatten())
    # print("labels[0]", labels[0].requires_grad)
    # return torch.stack(labels)

def predict_label(featmap, centroids, metric_function):
    scores = compute_negative_euclidean(featmap, centroids, metric_function)
    labels = []
    for idx in range(scores.size(0)):
        labels.append(scores[idx].topk(1, dim=0)[1].flatten().detach().cpu().numpy())
    return torch.tensor(labels)

def postprocess_label(args, K, idx, idx_img, scores, n_dual):
    out = scores[idx].topk(1, dim=0)[1].flatten().detach().cpu().numpy()

    # # Save labels. 
    # if not os.path.exists(os.path.join(args.save_model_path, 'label_' + str(n_dual))):
    #     os.makedirs(os.path.join(args.save_model_path, 'label_' + str(n_dual)))
    # torch.save(out, os.path.join(args.save_model_path, 'label_' + str(n_dual), '{}.pkl'.format(idx_img)))
    
    # Count for re-weighting. 
    counts = torch.tensor(np.bincount(out, minlength=K)).float()

    return counts


def eqv_transform_if_needed(args, dataloader, indice, input):
    if args.equiv:
        input = dataloader.dataset.transform_eqv(indice, input)

    return input  


def get_transform_params(args):
    inv_list = []
    eqv_list = []
    if args.augment:
        if args.blur:
            inv_list.append('blur')
        if args.grey:
            inv_list.append('grey')
        if args.jitter:
            inv_list.extend(['brightness', 'contrast', 'saturation', 'hue'])
        if args.equiv:
            if args.h_flip:
                eqv_list.append('h_flip')
            if args.v_flip:
                eqv_list.append('v_flip')
            if args.random_crop:
                eqv_list.append('random_crop')
    
    return inv_list, eqv_list


def collate_train(batch):
    if batch[0][-1] is not None:
        indice = [b[0] for b in batch]
        image1 = torch.stack([b[1] for b in batch])
        image2 = torch.stack([b[2] for b in batch])
        label1 = torch.stack([b[3] for b in batch])
        label2 = torch.stack([b[4] for b in batch])

        return indice, image1, image2, label1, label2
    
    indice = [b[0] for b in batch]
    image1 = torch.stack([b[1] for b in batch])

    return indice, image1

def collate_eval(batch):
    indice = [b[0] for b in batch]
    image = torch.stack([b[1] for b in batch])
    label = torch.stack([b[2] for b in batch])

    return indice, image, label 

def collate_train_baseline(batch):
    if batch[0][-1] is not None:
        return collate_eval(batch)
    
    indice = [b[0] for b in batch]
    image  = torch.stack([b[1] for b in batch])

    return indice, image

# def get_dataset(args, mode, inv_list=[], eqv_list=[]):
#     if args.cityscapes:
#         if mode == 'train':
#             dataset = TrainCityscapes(args.data_root, labeldir=args.save_model_path, res1=args.res1, res2=args.res2, 
#                                       split='train', mode='compute', inv_list=inv_list, eqv_list=eqv_list, scale=(args.min_scale, 1))
#         elif mode == 'train_val':
#             dataset = EvalCityscapes(args.data_root, res=args.res, split='val', mode='test',
#                                      label_mode=args.label_mode, long_image=args.long_image)
#         elif mode == 'eval_val':
#             dataset = EvalCityscapes(args.data_root, res=args.res, split=args.val_type, 
#                                      mode='test', label_mode=args.label_mode, long_image=args.long_image, label=False)
#         elif mode == 'eval_test':
#             dataset = EvalCityscapes(args.data_root, res=args.res, split='val', mode='test',
#                                      label_mode=args.label_mode, long_image=args.long_image)
#     else:
#         if mode == 'train':
#             dataset = TrainCOCO(args.data_root, labeldir=args.save_model_path, split='train', mode='compute', res1=args.res1,
#                                 res2=args.res2, inv_list=inv_list, eqv_list=eqv_list, thing=args.thing, stuff=args.stuff,
#                                 scale=(args.min_scale, 1))
#         elif mode == 'train_val':
#             dataset = EvalCOCO(args.data_root, res=args.res, split='val', mode='test', stuff=args.stuff, thing=args.thing)
#         elif mode == 'eval_val':
#             dataset = EvalCOCO(args.data_root, res=args.res, split=args.val_type, mode='test', label=False)
#         elif mode == 'eval_test':
#             dataset = EvalCOCO(args.data_root, res=args.res, split='val', mode='test', stuff=args.stuff, thing=args.thing)
    
#     return dataset 