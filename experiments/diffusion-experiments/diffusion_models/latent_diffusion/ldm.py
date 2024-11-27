import numpy as np 
import torch 
import math 
import copy 
import random 
import enum 
from torch import nn 
import torch.nn.functional as F 
from typing import List, Callable, Optional, List, Dict 
from lightning.pytorch.core import LightningModule
import sys 

class LDM(LightningModule):
    def __init__(
        self,
        denoising_model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        auto_normalize: bool = True,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: int = 5,
        model_var_type: ModelVarType = ModelVarType.FIXED_SMALL,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        loss_type: LossType = LossType.MSE,
        rescale_timesteps: bool = False,
    ) -> None:
        super().__init__()