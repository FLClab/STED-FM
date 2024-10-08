import torch
import torchvision
import torch.nn as nn
import tqdm.notebook as tqdm
import torch.nn.functional as F
import matplotlib.gridspec as gridspec

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import ToTensor, Normalize, Resize
from torchvision import models
from tqdm.auto import trange

class RELAX(nn.Module):
    def __init__(self, x, f, num_batches, batch_size, verbose=False):
        super().__init__()

        self.device = x.device
        self.batch_size = batch_size
        self.shape = tuple(x.shape[2:])
        self.num_batches = num_batches
        self.pdist = nn.CosineSimilarity(dim=1)
        self.verbose = verbose

        self.range_func = range
        if self.verbose:
            self.range_func = trange

        self.x = x
        self.encoder = f

        self.h_star = f(x)

        self.R = torch.zeros(self.shape, device=self.device)
        self.U = torch.zeros(self.shape, device=self.device)

        self.sum_of_weights = (1e-10)*torch.ones(self.shape, device=self.device)

    def forward(self):

        for batch in self.range_func(self.num_batches):
            for masks in self.mask_generator():

                x_mask = self.x * masks
                h = self.encoder(x_mask)
                sims = self.pdist(self.h_star, h)

                for si, mi in zip(sims, masks.squeeze()):

                    W_prev = self.sum_of_weights
                    self.sum_of_weights += mi

                    R_prev = self.R.clone()
                    self.R = self.R + mi*(si-self.R) / self.sum_of_weights
                    self.U = self.U + (si-self.R) * (si-R_prev) * mi

        return None

    def importance(self):
        return self.R

    def uncertainty(self):
        return self.U / (self.sum_of_weights - 1)

    def mask_generator(self, num_cells=7, p=0.5, nsd=2):

        pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)

        grid = (torch.rand(self.batch_size, 1, *((num_cells,) * nsd), device=self.device) < p).float()
        grid_up = F.interpolate(grid, size=(self.shape), mode='bilinear', align_corners=False)
        grid_up = F.pad(grid_up, pad_size, mode='reflect')

        shift_x = torch.randint(0, num_cells, (self.batch_size,), device=self.device)
        shift_y = torch.randint(0, num_cells, (self.batch_size,), device=self.device)

        masks = torch.empty((self.batch_size, 1, self.shape[-2], self.shape[-1]), device=self.device)

        for bi in range(self.batch_size):
            masks[bi] = grid_up[bi, :,
                                shift_x[bi]:shift_x[bi] + self.shape[-2],
                                shift_y[bi]:shift_y[bi] + self.shape[-1]]

        yield masks

    def __repr__(self):
        return f"RELAX: {self.num_batches} batches of {self.batch_size} samples"

def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        torchvision.transforms.Normalize([0.5001, 0.5001, 0.5001], [0.0003, 0.0003, 0.0003])
    ])
    test_dataset = CreateDataset('/home/ulaval.ca/koles2/tutosimclr/simclr/test_data', num_samples={'actin': 261, 'tubulin': 60, 'CaMKII': 56, 'PSD95': 61}, transform=transform, apply_filter=True, classes=['actin', 'tubulin', 'CaMKII', 'PSD95'])


    model_name_list = ['SimCLR']
    model_list = [load_simclr]
    relax_list = []
    mask_bs = 100
    num_batches = 30

    for model_loader, model_name in zip(model_list, model_name_list):
        model = model_loader()
        for idx, (img, label) in enumerate(test_dataset): 
            x = img.unsqueeze(0)  
            image = img.permute(1, 2, 0).numpy()
            with torch.no_grad():
                relax = RELAX(x.to('cuda'), model, num_batches, mask_bs)
                relax.forward()
            relax_list.append(relax)

            def to_np(x):
                return x.cpu().detach().numpy()

            fig = plt.figure()
            rows = 1
            columns = 3

            fig.add_subplot(rows, columns, 1)
            plt.imshow(image[:, :, 1], cmap='hot')

            plt.axis('off')
            plt.title(f'Input - {label}')

            fig.add_subplot(rows, columns, 2)
            plt.imshow(image[:, :, 1], cmap='hot')
            im = plt.imshow(to_np(relax_list[idx].importance()), alpha=0.75, cmap='bwr')
            plt.axis('off')
            plt.title(f'Importance')

            fig.add_subplot(rows, columns, 3)
            plt.imshow(image[:, :, 1], cmap='hot')
            plt.imshow(to_np(relax_list[idx].uncertainty()), alpha=0.75, cmap='bwr')
            plt.axis('off')
            plt.title(f'Uncertainty')

            cbar_ax = fig.add_axes([0.98, 0.05, 0.01, 0.86])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_ticks([])


            plt.savefig(f'/home/ulaval.ca/koles2/tutosimclr/relax/relax_{idx}', bbox_inches='tight')
        


if __name__ == "__main__":
    main()




