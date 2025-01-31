import numpy as np
import torch

def denormalize(img: np.ndarray) -> np.ndarray:
    return (img + 1.0) / 2.0

class SaveBestModel:
    def __init__(self, save_dir: str, model_name: str, maximize: bool = False):

        self.maximize = maximize
        if maximize:
            self.best_val = float('-inf')
        else:
            self.best_val = float('inf')

        self.model_name = model_name
        self.save_dir = save_dir

    def __call__(self, current_val: float, epoch: int, model, optimizer, criterion):
        if self.maximize:
            new_best = current_val > self.best_val
        else:
            new_best = current_val < self.best_val

        if new_best:
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

