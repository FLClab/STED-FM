
import torch
import torchvision

from transformers import AutoImageProcessor, ConvNextModel
from lightly.loss import NTXentLoss
# from convnext import convnext_small

import sys
sys.path.insert(0, "../experiments/")

from modules.loss import NTXentLossWithClasses

# backbone = torchvision.models.convnext_small()
# Ignore the classification head as we only want the features.
# backbone.classifier = torch.nn.Identity()
# model = backbone

# model = convnext_small()
# model = ConvNextModel.from_pretrained("facebook/convnext-small-224")

from model_builder import get_base_model
from tqdm.auto import trange

model, cfg = get_base_model("mae-lightning-small", in_channels=1)
criterion = NTXentLoss(temperature=0.1, gather_distributed=True)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
model = model.to("cuda")

for _ in trange(100):

    X = torch.randn(32, 1, 224, 224)
    X = X.to("cuda")
    z0 = model.forward_encoder(X)[:, 0]

    X = torch.randn(32, 1, 224, 224)
    X = X.to("cuda")
    z1 = model.forward_encoder(X)[:, 0]

    # loss = criterion(z0, z1)

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
