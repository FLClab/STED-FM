

from typing import Dict, List, Optional, Tuple, Union

import numpy
import random
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.utils import IMAGENET_NORMALIZE

from skimage import filters

class SimCLRTransform(MultiViewTransform):
    """Implements the transformations for SimCLR [0, 1].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    Note that SimCLR v1 and v2 use the same data augmentations.

    - [0]: SimCLR v1, 2020, https://arxiv.org/abs/2002.05709
    - [1]: SimCLR v2, 2020, https://arxiv.org/abs/2006.10029

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of [tensor, tensor].

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value. For datasets with small images,
            such as CIFAR, it is recommended to set `cj_strenght` to 0.5.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        cj_gamma: float = 0.8,
        scale: Tuple[float, float] = (0.75, 1.25),
        minimal_foreground : float = 0.05,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
        gaussian_noise_mu: float = 0.,
        gaussian_noise_std: float = 0.25,
        gaussian_noise_prob : float = 0.5,
        poisson_noise_lambda : float = 0.5,
        poisson_noise_prob : float = 0.5        
    ):
        view_transform = SimCLRViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            cj_gamma=cj_gamma,
            minimal_foreground = minimal_foreground,
            scale=scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
            gaussian_noise_mu = gaussian_noise_mu,
            gaussian_noise_std = gaussian_noise_std,
            gaussian_noise_prob = gaussian_noise_prob,
            poisson_noise_lambda = poisson_noise_lambda,
            poisson_noise_prob = poisson_noise_prob
        )
        super().__init__(transforms=[view_transform, view_transform])


class SimCLRViewTransform:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        cj_gamma: float = 0.8,
        scale: Tuple[float, float] = (0.75, 1.25),
        minimal_foreground : float = 0.05,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
        gaussian_noise_mu: float = 0.,
        gaussian_noise_std: float = 0.25,
        gaussian_noise_prob : float = 0.5,
        poisson_noise_lambda : float = 0.5,
        poisson_noise_prob : float = 0.5        
    ):
        # color_jitter = T.ColorJitter(
        #     brightness=cj_strength * cj_bright,
        #     contrast=cj_strength * cj_contrast,
        #     saturation=cj_strength * cj_sat,
        #     hue=cj_strength * cj_hue,
        # )
        color_jitter = MicroscopyColorJitter(
            p=cj_prob, 
            brightness=cj_strength * cj_bright,
            gamma=cj_strength * cj_gamma
        )

        transform = [
            #T.ToTensor(),
            RandomResizedCropMinimumForeground(size=input_size, scale=scale, min_fg=minimal_foreground),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            T.RandomApply([color_jitter], p=cj_prob),
            PoissonNoise(p=poisson_noise_prob, _lambda=poisson_noise_lambda),
            GaussianNoise(p=gaussian_noise_prob, mu=gaussian_noise_mu, std=gaussian_noise_std),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed
        
class MicroscopyColorJitter(torch.nn.Module):
    def __init__(self, 
                 p:float, 
                 brightness:Union[float, Tuple[float, float]], 
                 gamma: Union[float, Tuple[float, float]], 
                 *args, **kwargs) -> None:
        super().__init__()

        self.p = p

        if isinstance(brightness, float):
            brightness = (-1 * brightness, brightness)
        self.brightness = brightness

        if isinstance(gamma, float):
            gamma = (-1 * gamma, gamma)
        self.gamma = gamma

        self.options = [
            self.brightness_adjust,
            self.gamma_adjust
        ]

    def brightness_adjust(self, tensor: Tensor) -> Tensor:
        scale = random.uniform(*self.brightness)
        return torch.clamp_(tensor * (1 + scale), 0, 1)
    
    def gamma_adjust(self, tensor: Tensor) -> Tensor:
        gamma = random.uniform(*self.gamma)
        return torch.clamp_(tensor ** (1 + gamma), 0, 1)    

    def forward(self, tensor: Tensor) -> Tensor:
        options = list(range(len(self.options)))
        random.shuffle(options)
        for i in options:
            if random.random() < self.p:
                tensor = self.options[i](tensor)
        return tensor

# class RandomResizedCropMinimumForeground(T.RandomResizedCrop):
#     def __init__(self, size, scale, min_fg=0.01, threshold=0.02) -> None:
#         super().__init__(size, scale)

#         self.min_fg = min_fg
#         self.max_tries = 10
#         self.threshold = threshold

#     def get_params(self, image, scale, ratio):
#         """
#         Reimplements the get_params method from torchvision.transforms.RandomResizedCrop
#         """
#         image_height, image_width = image.size()[1], image.size()[2]

#         s = random.uniform(*scale)
#         h = int(round(self.size[0] * s))
#         w = int(round(self.size[1] * s))

#         if w <= image_width and h <= image_height:
#             i = random.randint(0, image_width - w)
#             j = random.randint(0, image_height - h)
#             return i, j, h, w
#         w = min(image_height, image_width)
#         return 0, 0, w, w

#     def forward(self, img) -> Tensor:
#         """
#         Implements a random resized crop with a minimum foreground
#         """
#         for n in range(self.max_tries):
#             i, j, h, w = self.get_params(img, self.scale, self.ratio)
#             crop = img[:, j : j + w, i : i + w] > self.threshold
#             if crop.sum() > self.min_fg * torch.numel(crop):
#                 break

#         if n == self.max_tries - 1:
#             h, w = self.size

#             fg = img > self.threshold
#             mask = torch.zeros_like(fg)
#             mask[:, : -h, :-w] = 1
#             fg = mask & fg
#             argwhere = torch.argwhere(fg)
#             if len(argwhere) == 0:
#                 j, i = 0, 0
#             else:
#                 _, j, i = argwhere[random.randint(0, argwhere.size(0) - 1)]

#         return F.resized_crop(img, j, i, h, w, self.size, self.interpolation, antialias=self.antialias)

        # # Precompute the foreground mask
        # key = (img.size()[1], img.size()[2], *img[0, 0, :10].tolist())
        # if key in self.precomputed_fg:
        #     fg = self.precomputed_fg[key]
        # else:
        #     fg = torch.nn.functional.avg_pool2d(img, self.bin_size)
        #     threshold = torch.quantile(fg, 0.50)
        #     fg = fg > threshold
        #     self.precomputed_fg[key] = fg

        # # Sample a random crop
        # s = random.uniform(*self.scale)
        # h = int(round(self.size[0] / self.bin_size * s))
        # w = int(round(self.size[1] / self.bin_size * s))
        # if h > (fg.size()[1] - 1) or w > (fg.size()[2] - 1):
        #     h, w = int(self.size[0] / self.bin_size), int(self.size[1] / self.bin_size)

        # # Sample a random crop with minimum foreground
        # argwhere = torch.argwhere(fg[:, :fg.size()[1] - h - 1, :fg.size()[2] - w - 1] > 0)
        # if argwhere.size(0) == 0:
        #     i, j = 0, 0
        # else:
        #     idx = random.randint(0, argwhere.size(0) - 1)
        #     _, j, i = argwhere[idx]
        #     i, j = i.item(), j.item()

        # # Return the resized crop
        # return F.resized_crop(
        #     img, 
        #     j * self.bin_size, i * self.bin_size, h * self.bin_size, w * self.bin_size, 
        #     self.size, self.interpolation, antialias=self.antialias
        # )

class RandomResizedCropMinimumForeground(T.RandomResizedCrop):
    def __init__(self, size, scale, min_fg=0.1) -> None:
        super().__init__(size, scale)

        self.min_fg = min_fg
        self.max_tries = 10

    def forward(self, img) -> Tensor:
        """
        Implements a random resized crop with a minimum foreground
        """
        img_array = img.numpy()
        threshold = filters.threshold_otsu(img_array)
        fg = img > threshold
        for _ in range(self.max_tries):
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            crop = fg[:, j : j + w, i : i + w]
            if crop.sum() > self.min_fg * torch.numel(crop):
                break
        return F.resized_crop(img, j, i, h, w, self.size, self.interpolation, antialias=self.antialias)

class PoissonNoise(torch.nn.Module):
    def __init__(self, p: float, _lambda: float) -> None:
        super().__init__()
        self.p = p
        self._lambda = _lambda
    
    def forward(self, tensor : Tensor) -> Tensor:
        if random.random() < self.p:
            # 255 is used to mimic typical acquisitions; the maximum 
            # We divide by 255 since the images are normalized
            rates = torch.clamp_(tensor, 0, 1) * 255.
            return torch.poisson(rates) / 255.
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(lambda={})'.format(self._lambda)

class GaussianNoise(torch.nn.Module):
    def __init__(self, p: float, mu : float, std : float) -> None:
        super().__init__()
        self.p = p
        self.mu = mu
        self.std = std
    
    def forward(self, tensor : Tensor) -> Tensor:
        if random.random() < self.p:
            std = random.uniform(0, self.std)
            mu = random.uniform(0, self.mu)
            return tensor + torch.randn(tensor.size()) * std + mu
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mu, self.std)