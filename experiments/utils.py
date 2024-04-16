import torch
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("dark_background")

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
