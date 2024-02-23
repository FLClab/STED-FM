import numpy as np
import matplotlib.pyplot as plt 
import torch 

import matplotlib.pyplot as plt 
import argparse
from models.classifier import ClassificationHead
from utils.data_utils import load_theresa_proteins
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.training_utils import AverageMeter, SaveBestModel