
from typing import Sequence, Union, Dict, List

import numpy
import torch
from torch import distributed as torch_dist
from torch import nn

from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.utils import dist
from lightly.loss import NTXentLoss

def masked_cross_entropy(logits, mask, reduce_func=torch.mean):
    """
    Compute the cross-entropy loss between logits and labels using a mask.

    :param logits: Tensor of shape (batch_size, num_classes)
    :param mask: Tensor of shape (batch_size, num_classes)
    :param reduce_func: Function to reduce the loss to a scalar.

    :return: The masked cross-entropy loss.
    """
    exponentiated = torch.exp(logits)
    loss = -torch.log(torch.sum(exponentiated * mask, dim=1) / torch.sum(exponentiated, dim=1))
    return reduce_func(loss)

class NTXentLossWithClasses(NTXentLoss):
    def __init__(self, *args, **kwargs):
        super(NTXentLossWithClasses, self).__init__(*args, **kwargs)

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, metadata: List):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Contrastive Cross Entropy Loss value.

        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = super(NTXentLoss, self).forward(
            out1, update=out0.requires_grad
        )

        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # use negatives from memory bank
            negatives = negatives.to(device)

            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum("nc,nc->n", out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum("nc,ck->nk", out0, negatives)

            # set the labels to the first "class", i.e. sim_pos,
            # so that it is maximized in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)

        else:
            # user other samples from batch as negatives
            # and create diagonal mask that only selects similarities between
            # views of the same image
            if self.gather_distributed and dist.world_size() > 1:
                # gather hidden representations from other processes
                out0_large = torch.cat(dist.gather(out0), 0)
                out1_large = torch.cat(dist.gather(out1), 0)
                diag_mask = dist.eye_rank(batch_size, device=out0.device)
            else:
                # single process
                out0_large = out0
                out1_large = out1
                diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

            # calculate similiarities
            # here n = batch_size and m = batch_size * world_size
            # the resulting vectors have shape (n, m)
            logits_00 = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
            logits_01 = torch.einsum("nc,mc->nm", out0, out1_large) / self.temperature
            logits_10 = torch.einsum("nc,mc->nm", out1, out0_large) / self.temperature
            logits_11 = torch.einsum("nc,mc->nm", out1, out1_large) / self.temperature

            # remove simliarities between same views of the same image
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)

            # concatenate logits
            # the logits tensor in the end has shape (2*n, 2*m-1)
            logits_0100 = torch.cat([logits_01, logits_00], dim=1)
            logits_1011 = torch.cat([logits_10, logits_11], dim=1)
            logits = torch.cat([logits_0100, logits_1011], dim=0)

            # create labels
            if metadata is None:
                labels = torch.arange(batch_size, device=device, dtype=torch.long)
                if self.gather_distributed:
                    labels = labels + dist.rank() * batch_size
            else:
                # Gets metadata from all processes
                if self.gather_distributed and dist.world_size() > 1:
                    metadata_large = torch.cat(dist.gather(metadata), 0)
                else:
                    metadata_large = metadata

                # Computes the labels as the unique values
                labels_large = torch.unique(metadata_large, return_inverse=True)[1]
                labels_large = labels_large.clone().detach().to(device=device, dtype=torch.long)
                if self.gather_distributed:
                    labels = labels_large[dist.rank() * batch_size : dist.rank() * batch_size + batch_size]
                else:
                    labels = labels_large

                # Create mask for the cross-entropy loss using vectorized operations
                mask = labels_large == labels[:, None]
                mask = torch.cat([mask, mask[~diag_mask].view(batch_size, -1)], dim=1)
                mask = mask.repeat(2, 1)
                
                # Compute the masked cross-entropy loss
                loss = masked_cross_entropy(logits, mask)
                return loss

            labels = labels.repeat(2)

        loss = self.cross_entropy(logits, labels)

        return loss
