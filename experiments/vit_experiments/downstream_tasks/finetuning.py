import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import tqdm
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import lightly.models.utils
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.insert(0, "../../proteins_experiments")
from utils.data_utils import fewshot_loader
from utils.training_utils import AverageMeter, SaveBestModel 
sys.path.insert(1, "../models")
from classifier import MAEClassificationHead


parser = argparse.ArgumentParser()
parser.add_argument("--class-type", type=str, default='protein')
parser.add_argument("--pretraining", type=str, default='STED')
parser.add_argument("--freeze", action='store_true')
args = parser.parse_args()

SAVE_EXPR = "linear-probe" if args.freeze else "finetuned"

class LightlyMAE(torch.nn.Module):
    def __init__(self, vit, in_channels: int = 1) -> None:
        super().__init__()
        decoder_dim = 512
        self.mask_ratio = 0.0

        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            in_chans=in_channels,
            decoder_depth=1,
            decoder_num_heads=8,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward_encoder(self, images: torch.Tensor, idx_keep: bool=None) -> torch.Tensor:
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x: torch.Tensor, idx_keep: bool, idx_mask: bool) -> torch.Tensor:
        batch_size = x.shape[0]
        x_decode = self.decoder.embed(x)
        x_masked = lightly.models.utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = lightly.models.utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))
        x_decoded = self.decoder.decode(x_masked)
        x_pred = lightly.models.utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        idx_keep, idx_mask = lightly.models.utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
        patches = lightly.models.utils.patchify(images, self.patch_size)
        target = lightly.models.utils.get_at_index(patches, idx_mask-1)
        return x_pred, target
    
def load_model():
    if args.pretraining == "STED":
        vit = vit_small_patch16_224(in_chans=1)
        backbone = LightlyMAE(vit=vit)
        checkpoint = torch.load("../Datasets/FLCDataset/baselines/checkpoint-530.pth")
        backbone.load_state_dict(checkpoint['model'])
        model = MAEClassificationHead(
            backbone=backbone,
            feature_dim=384,
            num_classes=4,
            freeze=args.freeze,
            global_pool="avg"
        )
    elif args.pretraining == "ImageNet":
        vit = vit_small_patch16_224(in_chans=3, pretrained=True)
        backbone = LightlyMAE(vit=vit, in_channels=3)
        # No need to load any checkpoint into the full MAE because the Decoder is never used in fine-tuning
        # So only need to load the encoder checkpoint (pretrained weights)
        model = MAEClassificationHead(
            backbone=backbone, 
            feature_dim=384,
            num_classes=4,
            freeze=args.freeze,
            global_pool='avg'
        )
    else:
        exit(f"Pretraining on {args.pretraining} not supported.")
    return model

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

def validation_step(
        model,
        valid_loader,
        criterion,
        epoch,
        device
):
    model.eval()
    loss_meter = AverageMeter()
    big_correct = np.array([0] * (4+1))
    big_n = np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, proteins, conditions in tqdm(valid_loader, desc="Validation..."):
            labels = proteins if args.class_type == 'protein' else conditions
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions = model(imgs)
            loss = criterion(predictions, labels)
            loss_meter.update(loss.item())
            correct, n = compute_Nary_accuracy(predictions, labels)
            big_correct = big_correct + correct
            big_n = big_n + n
        accuracies = big_correct / big_n
        print("********* Validation metrics **********")
        print("Epoch {} validation loss = {:.3f} ({:.3f})".format(
            epoch + 1, loss_meter.val, loss_meter.avg))
        print("Overall accuracy = {:.3f}".format(accuracies[0]))
        for i in range(1, 4+1):
            acc = accuracies[i]
            print("Class {} accuracy = {:.3f}".format(
                i, acc))
    return loss_meter.avg, accuracies[0]

def train_one_epoch(
        train_loader, 
        model,
        optimizer,
        criterion, 
        device,
        epoch
):
    model.train()
    loss_meter = AverageMeter()
    for imgs, proteins, conditions in tqdm(train_loader, desc="Training...."):
        labels = proteins if args.class_type == "protein" else conditions
        imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    print("Epoch {} loss = {:.3f} ({:.3f})".format(
        epoch + 1, loss_meter.val, loss_meter.avg))
    # print(f"The number of images sampled per class for this epoch was: {class_counts}")
    return loss_meter.avg

def train(
        model,
        train_loader, 
        valid_loader,
        device,
        num_epochs, 
        criterion, 
        optimizer, 
        scheduler, 
        model_path
): 
    train_loss, val_loss, val_acc, lrates = [], [], [], []
    save_best_model = SaveBestModel(
        save_dir=model_path,
        model_name=f"{SAVE_EXPR}_model",
    )
    for epoch in tqdm(range(num_epochs), desc="Epochs..."):
        loss = train_one_epoch(
            train_loader=train_loader,
            model=model, 
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            device=device
            )
        train_loss.append(loss)
        v_loss, v_acc = validation_step(model, valid_loader, criterion, epoch, device)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        scheduler.step()
        temp_lr = optimizer.param_groups[0]['lr']
        lrates.append(temp_lr)
        save_best_model(v_loss, epoch=epoch, model=model, optimizer=optimizer, criterion=criterion)
        plot_training_curves(train_loss, val_loss, val_acc, lrates, model_path)

def plot_training_curves(train_loss, val_loss, val_acc, learning_rates, model_path):
    fig, axs = plt.subplots(2, 1, sharex=True)
    x = np.arange(0, len(train_loss), 1)
    ax1 = axs[0].twinx()
    ax2 = axs[0].twinx()
    axs[0].plot(x, train_loss, color='lightblue', label="Train")
    axs[0].plot(x, val_loss, color='lightcoral', label="Validation")
    ax1.plot(x, learning_rates, color='lightgreen', label='lr', ls='--')
    axs[1].plot(x, val_acc, color='lightcoral', label="Validation")
    axs[1].set_xlabel('Epochs')
    ax2.plot(x, learning_rates, color='palegreen', label='lr', ls='--', alpha=0.1)
    axs[1].set_ylabel('Accuracy')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].legend()
    fig.savefig(f"{model_path}/{SAVE_EXPR}_curves.png")
    plt.close(fig)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    train_loader, valid_loader, test_loader = fewshot_loader(
        path="../Datasets/FLCDataset/TheresaProteins",
        class_type='protein',
        n_channels=1 if args.pretraining == "STED" else 3
    )
    model = load_model().to(device)
    if args.freeze:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=20)
    criterion = torch.nn.CrossEntropyLoss()
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=500,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/vit_experiments/Datasets/FLCDataset/baselines/finetuning/{args.pretraining}"
    )
    

if __name__=="__main__":
    main()