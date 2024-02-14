import torch
import matplotlib.pyplot as plt
import numpy as np

def track_loss(train_loss: list, lrates: list, save_dir: str) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    x = np.arange(0, len(train_loss), 1)
    ax1.plot(x, train_loss, color='steelblue', label="Train")
    # ax1.plot(x, val_loss, color='firebrick', label='Validation')
    ax2.plot(x, lrates, color='forestgreen', label='LR', ls='--')
    ax1.legend()
    ax2.legend()
    # ax1.set_yscale('log')
    fig.savefig(f"{save_dir}/training_curves.png")
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
