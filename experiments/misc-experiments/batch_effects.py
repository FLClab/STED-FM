
import random
import torch
import numpy

class BaseBatchEffect:
    @property
    def name(self):
        return self.__class__.__name__
    
    def apply(self, X):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name

class IdendityBatchEffect(BaseBatchEffect):
    @property
    def name(self):
        return "IdendityBatchEffect()"

    def apply(self, X):
        return X

class PoissonBatchEffect(BaseBatchEffect):
    def __init__(self, _lambda):
        self._lambda = _lambda

    @property
    def name(self):
        return f"PoissonBatchEffect({self._lambda})"

    def apply(self, X):
        noise = torch.poisson(torch.ones_like(X) * self._lambda) / 255.0
        return X + noise
    
class GaussianBatchEffect(BaseBatchEffect):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @property
    def name(self):
        return f"GaussianBatchEffect({self.mean}, {self.std})"

    def apply(self, X):
        noise = torch.normal(mean=self.mean, std=self.std, size=X.size())
        if X.size(0) == 1:
            return torch.clamp(X + noise, 0, None)
        return X + noise
    
class ContrastBatchEffect(BaseBatchEffect):
    def __init__(self, cutoff, gain=10):
        self.cutoff = cutoff
        self.gain = gain

    @property
    def name(self):
        return f"ContrastBatchEffect({self.cutoff}, {self.gain})"

    def apply(self, X):
        vmin, vmax = X.min(), X.max()
        X = (X - vmin) / (vmax - vmin)
        output = 1 / (1 + torch.exp(self.gain * (self.cutoff - X)))
        return torch.clamp(output * (vmax - vmin) + vmin, vmin, vmax)

class RotationBatchEffect(BaseBatchEffect):
    @property
    def name(self):
        return f"RotationBatchEffect()"

    def apply(self, X):
        dims = [1, 2]
        if X.ndim == 4:
            dims = [2, 3]
        num = random.randint(1, 3)
        return torch.rot90(X, num, dims)
    
class FlipBatchEffect(BaseBatchEffect):
    @property
    def name(self):
        return f"FlipBatchEffect()"

    def apply(self, X):
        dims = [1, 2]
        if X.ndim == 4:
            dims = [2, 3]
        choice = random.choice(dims)
        return torch.flip(X, [choice])
        
class ScaleBatchEffect(BaseBatchEffect):
    def __init__(self, scale):
        self.scale = scale

    @property
    def name(self):
        return f"ScaleBatchEffect({self.scale})"

    def apply(self, X):
        return X * self.scale
    
class OffsetBatchEffect(BaseBatchEffect):
    def __init__(self, offset):
        self.offset = offset

    @property
    def name(self):
        return f"OffsetBatchEffect({self.offset})"

    def apply(self, X):
        return X + self.offset
    
class GaussianBlurBatchEffect(BaseBatchEffect):
    def __init__(self, sigma):
        self.sigma = sigma

    @property
    def name(self):
        return f"GaussianBlurBatchEffect({self.sigma})"

    def apply(self, X):
        return gaussian_blur(X, kernel_size=5, sigma=self.sigma)