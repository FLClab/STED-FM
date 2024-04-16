import torch
import numpy

class IntensityModel(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x):
        x = x.cpu().numpy()
        quantiles = numpy.quantile(x, q=[0., 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], axis=(1, 2, 3)).T
        quantiles = torch.tensor(quantiles)

        return quantiles
    
def get_intensity_model(name: str = "naive") -> torch.nn.Module:
    if name == "naive":
        backbone = IntensityModel()
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone