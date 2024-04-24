import io
import os
import matplotlib 
import pickle
import torch
import argparse
import json
import dataclasses
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

def update_cfg(cfg: dataclass, opts: list[str]) -> dataclass:
    """
    Updates the configuration with additional options inplace

    :param cfg: A `dataclass` of the configuration
    :param opts: A `list` of options to update the configuration
    """
    for i in range(0, len(opts), 2):
        key, value = opts[i], opts[i + 1]
        if len(key.split(".")) > 1:
            key, subkey = key.split(".")
            update_cfg(getattr(cfg, key), [subkey, value])
        else:
            if type(getattr(cfg, key)) == bool:
                # Special case for boolean values
                setattr(cfg, key, value in ("True", "true", "1"))
            else:
                setattr(cfg, key, type(getattr(cfg, key))(value))

def save_cfg(cfg: dataclass, path: str):
    """
    Saves the configuration to a file

    :param cfg: A `dataclass` of the configuration
    :param path: A `str` of the path to save the configuration
    """
    out = {}
    for key, value in cfg.__dict__.items():
        if dataclasses.is_dataclass(value):
            out[key] = save_cfg(value, None)
        elif isinstance(value, argparse.Namespace):
            out[key] = value.__dict__
        else:
            out[key] = value

    # Save to file; if path is None, return the dictionary for recursive calls
    if isinstance(path, str):
        json.dump(out, open(path, "w"), indent=4, sort_keys=True)
    return out

def savefig(fig, savepath, extension="pdf", save_white=False, **kwargs):
    """
    Utilitary function allowing to save the figure to 
    the savepath
    
    :param fig: A `matplotlib.Figure`
    :param ax: A `matplotlib.Axes`  
    :param savepath: A `str` of the filename
    :param extension: A `str` of the extension of the file
    :param save_white: A `bool` wheter to save the figure in white version 
                       as well
    """
    dirname = os.path.dirname(savepath)
    basename = os.path.basename(savepath)    
    
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    
    fig.savefig(f"{savepath}.{extension}", bbox_inches="tight", transparent=True)
    if save_white:
        
        # Creates empty directory
        os.makedirs(os.path.join(dirname, "white"), exist_ok=True)
        savepath = os.path.join(dirname, "white", basename)

        buf = io.BytesIO()
        pickle.dump(fig, buf)
        buf.seek(0)
        fig = pickle.load(buf)
        
        change_figax_color(fig, **kwargs)
        fig.savefig(f"{savepath}.{extension}", bbox_inches="tight", transparent=True, dpi=600)
        
        plt.close(fig)
        
def change_figax_color(fig):
    """
    Utilitary function allowing to change the figure and 
    ax color from black to white
    
    :param fig: A `matplotlib.Figure`
    :param ax: A `matplotlib.Axes`    
    """
    def _change_ax(ax):
        ax.set_facecolor("none")
        for child in ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('white')      
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color("white")
        
        # For line plots
        for line in ax.get_lines():
            if line.get_color() in ["#000000", "000000", "black"]:
                line.set_color("white")    

        # For scatter plots
        for collection in ax.collections:
            new_colors = ["white" if matplotlib.colors.to_hex(c) == "#000000" else c 
                             for c in collection.get_facecolors()]
            new_colors = [mimic_white_alpha(c) for c in new_colors]
            collection.set_facecolors(new_colors)
            collection.set_alpha(1)
            new_colors = ["white" if matplotlib.colors.to_hex(c) == "#000000" else c 
                             for c in collection.get_edgecolors()] 
            new_colors = [mimic_white_alpha(c) for c in new_colors]            
            collection.set_edgecolors(new_colors)
            collection.set_alpha(1)            

        # For hist plots
        for patch in ax.patches:
            c = patch.get_facecolor()
            if matplotlib.colors.to_hex(c) == "#000000":
                patch.set_color("white")        
        
    # Change figure background
    fig.patch.set_facecolor("none")
    
    # Changes colorbars if any
    for ax in fig.axes:
        _change_ax(ax.axes)
        
def mimic_white_alpha(color):
    """
    Mimics the color that would be perceived of a color on a white 
    background
    
    :param color: A `matplotlib.collections` of lines
    
    :returns : A `list` of the colors
    """
    c_rgba = matplotlib.colors.to_rgba(color)
    c_rgb, alpha = c_rgba[:3], c_rgba[-1]
    return matplotlib.colors.to_hex(tuple(c * alpha + (1 - alpha) for c in c_rgb))  

def apply_alpha(c, alpha):
    """
    Applies an alpha value to a color
    
    :param c: A `tuple` of the current color
    """
    newc = [x for x in c]
    newc[-1] = alpha
    return mimic_white_alpha(newc)

def compute_Nary_accuracy(preds: torch.Tensor, labels: torch.Tensor, N: int = 4) -> list:
    # accuracies = []
    correct = []
    big_n = []
    _, preds = torch.max(preds, 1)
    assert preds.shape == labels.shape
    c = torch.sum(preds == labels)
    correct.append(c.item())
    big_n.append(preds.shape[0])
    for n in range(N):
        c = ((preds == labels ) * (labels == n)).float().sum().cpu().detach().numpy()
        n = (labels==n).float().sum().cpu().detach().numpy()
        correct.append(c)
        big_n.append(n)
        # temp = ( (preds == labels) * (labels == n)).float().sum() / (labels == n).float().sum()
        # accuracies.append(temp.cpu().detach().numpy())
    return np.array(correct), np.array(big_n)

def track_loss(train_loss: list, val_loss: list, val_acc: list, lrates: list, save_dir: str) -> None:
    fig, axs = plt.subplots(2, 1, sharex=True)
    x = np.arange(0, len(train_loss), 1)
    ax1 = axs[0].twinx()
    ax2 = axs[0].twinx()
    axs[0].plot(x, train_loss, color='lightblue', label="Train")
    axs[0].plot(x, val_loss, color='lightcoral', label="Validation")
    ax1.plot(x, lrates, color='lightgreen', label='lr', ls='--')
    axs[1].plot(x, val_acc, color='lightcoral', label="Validation")
    axs[1].set_xlabel('Epochs')
    ax2.plot(x, lrates, color='palegreen', label='lr', ls='--', alpha=0.1)
    axs[1].set_ylabel('Accuracy')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].legend()
    fig.savefig(f"{save_dir}")
    plt.close(fig)

class SaveBestModel:
    def __init__(self, save_dir: str, model_name: str, best_val: float = float('inf')):
        self.best_val = best_val
        self.model_name = model_name
        self.save_dir = save_dir

    def __call__(self, current_val: float, epoch: int, model, optimizer, criterion):
        if current_val < self.best_val:
            self.best_val = current_val
            print(f"Best loss: {self.best_val}")
            print(f"Saving best model for epoch: {epoch + 1} at {self.save_dir}\n")
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                }, '{}/{}.pth'.format(self.save_dir, self.model_name))
            
class AverageMeter:
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
