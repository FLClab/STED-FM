import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def freeze_select_params(model, expr):
    for n, p in model.named_parameters():
        if expr in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def sanity_check(imgs: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> None:
    indices = np.arange(0, imgs.shape[0], 1)
    for i, img, label, mask in zip(indices, imgs, labels, masks):
        img = img.squeeze(0).cpu().detach().numpy()
        mask = mask.squeeze(0).cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img, cmap='hot')
        axs[0].set_title(label)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.savefig("./to_delete/sanity_check_{}.png".format(i))
        plt.close(fig)


class SaveBestModel:
    def __init__(self, model_name, save_dir, best: float = float('inf'), maximize: bool = False) -> None:
        self.best = best
        self.maximize = maximize
        self.save_dir = save_dir
        self.model_name = model_name

    def __call__(self, current, epoch, model, optimizer, criterion=None):
        if self.maximize:
            if current > self.best:
                self.best = current
                print(
                    f"Saving best model for epoch {epoch + 1} at {self.save_dir}\n")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                }, f"{self.save_dir}/{self.model_name}.pth")
        else:
            if current < self.best:
                self.best = current
                print(
                    f"Saving best model for epoch {epoch + 1} at {self.save_dir}\n")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                }, f"{self.save_dir}/{self.model_name}.pth")


def inspect_loss(train_loss, val_loss, save_dir="./checkpoints"):
    fig = plt.figure()
    epochs = np.arange(0, len(train_loss), 1)
    plt.plot(epochs, train_loss, color='lightblue', label="Train", marker='.')
    plt.plot(epochs, val_loss, color='lightcoral',
             label="Validation", marker='.')
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    fig.savefig(f"{save_dir}/loss_figure.png")


class AverageMeter:
    def __init__(self) -> None:
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
