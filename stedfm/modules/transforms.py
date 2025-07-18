

from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy
import random
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform, RandomRotate, RandomRotateDegrees
from lightly.transforms.utils import IMAGENET_NORMALIZE

from skimage import filters

class PiCIETransform(MultiViewTransform):
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
        view_transform = PiCIEViewTransform(
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
        self.initial_transform = T.Compose([
            T.ToTensor(),
            RandomResizedCropMinimumForeground(size=int(input_size * 1.25), scale=(1.0, 1.0), min_fg=minimal_foreground)
        ])
        super().__init__(transforms=[view_transform, view_transform])

    def __call__(self, image: Union[Tensor, Image]) -> Union[List[Tensor], List[Image]]:
        """Transforms an image into multiple views.

        Every transform in self.transforms creates a new view.

        Args:
            image:
                Image to be transformed into multiple views.

        Returns:
            List of views.

        """
        image = self.initial_transform(image)
        return [transform(image) for transform in self.transforms]        

class ReversibleRandomRotate:
    """Implementation of random rotation.

    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.

    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.

    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing
            any artifacts.

    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = angle

    def __call__(self, image: Union[Image, Tensor], transformations) -> Union[Image, Tensor]:
        """Rotates the image with a given probability.

        Args:
            image:
                PIL image or tensor which will be rotated.

        Returns:
            Rotated image or original image.

        """
        prob = numpy.random.random_sample()
        rotation = 0
        if prob < self.prob:
            image = F.rotate(image, self.angle)
            rotation = self.angle
        transformations["rotation"] = rotation
        return image

def reversible_random_rotation_transform(rr_prob, rr_degrees):
    if rr_degrees is None:
        # Random rotation by 90 degrees.
        return ReversibleRandomRotate(prob=rr_prob, angle=90)
    else:
        raise NotImplementedError("Reversible random rotation by arbitrary degrees is not implemented yet.")

class ReversibleHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.prob = p

    def __call__(self, image: Union[Image, Tensor], transformations) -> Union[Image, Tensor]:
        prob = numpy.random.random_sample()
        hflip = 1
        if prob < self.prob:
            image = F.hflip(image)
            hflip = -1
        transformations["hflip"] = hflip
        return image

class ReversibleVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.prob = p

    def __call__(self, image: Union[Image, Tensor], transformations) -> Union[Image, Tensor]:
        prob = numpy.random.random_sample()
        vflip = 1
        if prob < self.prob:
            image = F.vflip(image)
            vflip = -1
        transformations["vflip"] = vflip
        return image

class MultiArgumentCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args, **kwargs):
        for transform in self.transforms:
            image = transform(image, *args, **kwargs)
        return image
    
def batch_apply_transforms(func: Callable, images: List[Union[Image, Tensor]], transformations: Dict[str, Tensor]) -> List[Union[Image, Tensor]]:
    output=[]
    for i, image in enumerate(images):
        t = {key : value[i].item() for key, value in transformations.items()}
        output.append(func(image, t))
    return torch.stack(output)

def apply_inverse_transforms(image: Union[Image, Tensor], transformations: Dict[str, int]) -> Union[Image, Tensor]:
    if "hflip" in transformations:
        image = F.hflip(image) if transformations["hflip"] == -1 else image
    if "vflip" in transformations:
        image = F.vflip(image) if transformations["vflip"] == -1 else image
    if "rotation" in transformations:
        image = F.rotate(image, -transformations["rotation"]) if transformations["rotation"] != 0 else image
    return image

def apply_forward_transforms(image: Union[Image, Tensor], transformations: Dict[str, int]) -> Union[Image, Tensor]:
    if "rotation" in transformations:
        image = F.rotate(image, transformations["rotation"]) if transformations["rotation"] != 0 else image
    if "vflip" in transformations:
        image = F.vflip(image) if transformations["vflip"] == -1 else image
    if "hflip" in transformations:
        image = F.hflip(image) if transformations["hflip"] == -1 else image
    return image

class PiCIEViewTransform:
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
        self.crop_transform = T.Compose([
            RandomResizedCropMinimumForeground(size=input_size, scale=scale, min_fg=minimal_foreground, return_bbox=True)
        ])
        self.geometric_transform = MultiArgumentCompose([
            reversible_random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            ReversibleHorizontalFlip(p=hf_prob),
            ReversibleVerticalFlip(p=vf_prob)
        ])
        transform = [
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            T.RandomApply([color_jitter], p=cj_prob),
            PoissonNoise(p=poisson_noise_prob, _lambda=poisson_noise_lambda),
            GaussianNoise(p=gaussian_noise_prob, mu=gaussian_noise_mu, std=gaussian_noise_std),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.invariance_transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        image, bbox = self.crop_transform(image)
        transformations = {}
        image = self.geometric_transform(image, transformations)
        transformed: Tensor = self.invariance_transform(image)
        bbox = torch.tensor(bbox)
        return [transformed, bbox, transformations]

class DINOTransform(MultiViewTransform):
    """Implements the global and local view augmentations for DINO [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2 * global + n_local_views. (8 by default)

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Random solarization
        - ImageNet normalization

    This class generates two global and a user defined number of local views
    for each image in a batch. The code is adapted from [1].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        global_crop_size:
            Crop size of the global views.
        global_crop_scale:
            Tuple of min and max scales relative to global_crop_size.
        local_crop_size:
            Crop size of the local views.
        local_crop_scale:
            Tuple of min and max scales relative to local_crop_size.
        n_local_views:
            Number of generated local views.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Tuple of probabilities to apply gaussian blur on the different
            views. The input is ordered as follows:
            (global_view_0, global_view_1, local_views)
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        kernel_scale:
            Old argument. Value is deprecated in favor of sigmas. If set, the old behavior applies and `sigmas` is ignored.
            Used to scale the `kernel_size` of a factor of `kernel_scale`
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        solarization:
            Probability to apply solarization on the second global view.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_size: int = 96,
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        n_local_views: int = 6,
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
        # first global crop
        global_transform_0 = SimCLRViewTransform(
            input_size=global_crop_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            cj_gamma=cj_gamma,
            minimal_foreground = minimal_foreground,
            scale=global_crop_scale,
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

        # second global crop
        global_transform_1 = SimCLRViewTransform(
            input_size=global_crop_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            cj_gamma=cj_gamma,
            minimal_foreground = minimal_foreground,
            scale=global_crop_scale,
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

        # transformation for the local small crops
        local_transform = SimCLRViewTransform(
            input_size=local_crop_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            cj_gamma=cj_gamma,
            minimal_foreground = minimal_foreground,
            scale=local_crop_scale,
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
        local_transforms = [local_transform] * n_local_views
        transforms = [global_transform_0, global_transform_1]
        transforms.extend(local_transforms)
        super().__init__(transforms)

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
            T.ToTensor(),
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
        return torch.clamp(tensor * (1 + scale), 0, 1)
    
    def gamma_adjust(self, tensor: Tensor) -> Tensor:
        gamma = random.uniform(*self.gamma)
        return torch.clamp(tensor ** (1 + gamma), 0, 1)    

    def forward(self, tensor: Tensor) -> Tensor:
        options = list(range(len(self.options)))
        random.shuffle(options)
        for i in options:
            if random.random() < self.p:
                tensor = self.options[i](tensor)
        return tensor

class RandomResizedCropMinimumForeground(T.RandomResizedCrop):
    def __init__(self, size, scale, min_fg=0.01, threshold=0.02, return_bbox=False) -> None:
        super().__init__(size, scale)

        self.min_fg = min_fg
        self.max_tries = 10
        self.threshold = threshold
        self.return_bbox = return_bbox

    def get_params(self, image, scale, ratio):
        """
        Reimplements the get_params method from torchvision.transforms.RandomResizedCrop
        """
        image_height, image_width = image.size()[1], image.size()[2]

        s = random.uniform(*scale)
        h = int(round(self.size[0] * s))
        w = int(round(self.size[1] * s))

        if w <= image_width and h <= image_height:
            i = random.randint(0, image_width - w)
            j = random.randint(0, image_height - h)
            return i, j, h, w
        w = min(image_height, image_width)
        return 0, 0, w, w

    def forward(self, img) -> Tensor:
        """
        Implements a random resized crop with a minimum foreground
        """
        h, w = self.size
        image_height, image_width = img.size()[1], img.size()[2]
        if h >= image_height and w >= image_width:
            if self.return_bbox:
                return img, (0, 0, image_height, image_width)
            return img
                
        for n in range(self.max_tries):
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            crop = img[:, j : j + w, i : i + w] > self.threshold
            if crop.sum() > self.min_fg * torch.numel(crop):
                break
        
        if n == self.max_tries - 1:
            h, w = self.size

            fg = img > self.threshold
            mask = torch.zeros_like(fg)
            mask[:, h // 2 : -h // 2, w // 2 : -w // 2] = 1
            fg = mask & fg
            argwhere = torch.argwhere(fg)
            if len(argwhere) == 0:
                j, i = 0, 0
            else:
                _, j, i = argwhere[random.randint(0, argwhere.size(0) - 1)]
                j = j - h // 2
                i = i - w // 2

        if self.return_bbox:
            return F.resized_crop(img, j, i, h, w, self.size, self.interpolation, antialias=self.antialias), (j, i, h, w)
        return F.resized_crop(img, j, i, h, w, self.size, self.interpolation, antialias=self.antialias)

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

# class RandomResizedCropMinimumForeground(T.RandomResizedCrop):
#     def __init__(self, size, scale, min_fg=0.1) -> None:
#         super().__init__(size, scale)

#         self.min_fg = min_fg
#         self.max_tries = 10

#     def forward(self, img) -> Tensor:
#         """
#         Implements a random resized crop with a minimum foreground
#         """
#         img_array = img.numpy()
#         threshold = filters.threshold_otsu(img_array)
#         fg = img > threshold
#         for _ in range(self.max_tries):
#             i, j, h, w = self.get_params(img, self.scale, self.ratio)
#             crop = fg[:, j : j + w, i : i + w]
#             if crop.sum() > self.min_fg * torch.numel(crop):
#                 break
#         return F.resized_crop(img, j, i, h, w, self.size, self.interpolation, antialias=self.antialias)

class PoissonNoise(torch.nn.Module):
    def __init__(self, p: float, _lambda: float) -> None:
        super().__init__()
        self.p = p
        self._lambda = _lambda
    
    def forward(self, tensor : Tensor) -> Tensor:
        if random.random() < self.p:
            # 255 is used to mimic typical acquisitions; the maximum 
            # We divide by 255 since the images are normalized
            rates = torch.clamp(tensor, 0, 1) * 255.
            return torch.clamp(torch.poisson(rates) / 255., 0, 1)
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