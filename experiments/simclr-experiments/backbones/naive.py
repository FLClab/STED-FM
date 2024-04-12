
import torch
import numpy

class NaiveModel(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x):
        x = x.cpu().data.numpy()
        quantiles = numpy.quantile(x, q=[0., 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], axis=(1, 2, 3)).T
        quantiles = torch.tensor(quantiles)

        return quantiles
    
def get_backbone(name: str) -> torch.nn.Module:
    if name == "naive":
        backbone = NaiveModel()
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone