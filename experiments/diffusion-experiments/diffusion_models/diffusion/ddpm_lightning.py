import numpy as np
import torch 
import math 
import copy 
import random 
import enum
from torch import nn 
import torch.nn.functional as F 
from typing import List, Callable, Optional, Dict, List
from .denoising.unet import UNet
from lightning.pytorch.core import LightningModule
import sys 

class_dict = {
    "FUS": 0,
    "adducin": 1,
    "bassoon": 2,
    "beta-camkii": 3,
    "beta2-spectrin": 4,
    "camkii": 5,
    "f-actin": 6,
    "gephyrin": 7,
    "glur1": 8,
    "homer": 9,
    "lifeact": 10,
    "live-tubulin": 11,
    "map2": 12, 
    "nkcc2": 13,
    "psd95": 14,
    "rim": 15,
    "sir-actin": 16,
    "sir-tubulin": 17,
    "tom20": 18,
    "tubulin": 19,
    "vgat": 20,
    "vglut1": 21,
    "vglut2": 22,
    "vimentin": 23,
}

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DDPM(LightningModule):
    def __init__(
        self,
        denoising_model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "sigmoid",
        auto_normalize: bool = True,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: int = 5,
        model_var_type: ModelVarType = ModelVarType.FIXED_SMALL,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        loss_type: LossType = LossType.MSE,
        rescale_timesteps: bool = False,
        condition_type: Optional[str] = None,
        latent_encoder: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.condition_type = condition_type
        self.rescale_timesteps = rescale_timesteps
        self.model_var_type = model_var_type
        self.model_mean_type = model_mean_type 
        self.loss_type = loss_type
        self.model = denoising_model 
        self.latent_encoder = latent_encoder
        if self.latent_encoder is not None:
            for p in self.latent_encoder.parameters():
                p.requires_grad = False
        self.T = timesteps 
        self.channels = self.model.channels 
        betas = get_named_beta_schedule(schedule_name=beta_schedule, num_diffusion_timesteps=self.T)
        self.betas = betas 
        assert len(betas.shape) == 1 
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.T,)

        # q(x_t, x_{t-1})
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def q_mean_variance(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get the distribution q(x_t, x_0)
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = _extractor_into_tensor(1.0 - self.alphas_cumprod, t, x_0.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance 

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Diffuse the data, i.e., sample from q(x_t, x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape 
        return(
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape 
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_0
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_0.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, cond=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = self.model(x, self._scale_timesteps(t), cond=cond, **model_kwargs)

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_x0 = process_xstart(
            self._predict_x0_from_eps(x_t=x, t=t, eps=model_output)
        )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_0=pred_x0, x_t=x, t=t
        )
        assert (model_mean.shape == model_log_variance.shape == pred_x0.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x0": pred_x0,
        }

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        assert x_t.shape == eps.shape 
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t 
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_x0(self, x_t, t, pred_x0):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_x0
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable] = None,
        model_kwargs=None,
        cond=None
    ) -> torch.Tensor:
        """
        Sample x_{t-1}
        """
        out = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            cond=cond,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t!=0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x0": out["pred_x0"]}

    def forward_backward(self, x_0: torch.Tensor, t_fixed: int = None, clip_denoised: bool=True):
        if t_fixed == 0:
            return x_0.detach()
        
        if t_fixed is None:
            t_fixed = self.T 

        sequence = [x_0.cpu().detach()]
        t_start = torch.tensor([t_fixed - 1], device=x_0.device).repeat(x_0.shape[0])
        x_t = self.q_sample(x_0=x_0, t=t_start)
        noisy_img = copy.deepcopy(x_t)
        for t in range(int(t_fixed - 1), -1, -1):
            t = torch.tensor([t], device=x_0.device).repeat(x_0.shape[0])
            with torch.no_grad():
                out = self.p_sample(x=x_t, t=t, clip_denoised=clip_denoised)
                x_t = out["sample"]
        return x_t, noisy_img

    def ddim_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable] = None,
        model_kwargs=None,
        cond_fn=None,
        eta: float = 0.0,
        cond=None,
    ) -> torch.Tensor:
        out = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            cond=cond,
        )
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta * torch.sqrt( (1 - alpha_bar_prev) / (1 - alpha_bar) ) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        eps = self._predict_eps_from_x0(x, t, out["pred_x0"])
        
        mean_pred = out["pred_x0"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps 
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))) 
        sample = mean_pred + nonzero_mask * sigma * noise 
        return {"sample": sample, "pred_x0": out["pred_x0"]}


    def ddim_sample_loop(
        self,
        shape, 
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta: float = 0.0,
        cond=None
    ):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            shape, 
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            cond=cond,
        ):
            final = sample
        return final["sample"]


    def ddim_sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        cond=None,
    ):
        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise 
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.T))[::-1] 

        if progress:
            from tqdm.auto import tqdm 
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    img, 
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    cond=cond
                )            
                yield out 
                img = out["sample"]

    def ddim_reverse_sample(
        self,
        x, 
        t, 
        clip_denoised=True,
        denoised_fn=True,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Use this function to sample x_{t+1} in a deterministic manner
        """
        assert eta == 0.0
        out = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            cond=None,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_x0"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = (out["pred_x0"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps)
        return {"sample": mean_pred, "pred_x0": out["pred_x0"]}

    def ddim_reverse_sample_loop(
        self,
        x: torch.Tensor,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0, 
        device=None,
    ):
        if device is None:
            device = next(self.model.parameters()).device
        sample_t = []
        x0_t = []
        T = []
        indices = list(range(self.T))
        sample = x 
        for i in indices:
            t = torch.tensor([i] * x.shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(
                    x=sample,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta
                )
                sample = out["sample"]
                sample_t.append(sample)
                x0_t.append(out["pred_x0"])
                T.append(t)
        return {
            "sample": sample,
            "sample_t": sample_t,
            "x0_t": x0_t,
            "T": T,
        }

    def p_sample_loop(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        cond=None,
    ):
        """
        Generate samples from the model.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            cond=cond,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        cond=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.T))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, desc="Iterative sampling...")

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    cond=cond,
                )
                yield out
                img = out["sample"]


    def _vb_terms_bpd(
        self, x_0, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_0=x_0, x_t=x_t, t=t
        )
        out = self.p_mean_variance( x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_0.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def forward(self, x_0, t, cond: torch.Tensor = None, model_kwargs=None, noise=None, verbose=True):
        """
        Compute training losses for a single timestep.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)

        # latent_encoding = self.latent_encoder(x0)
        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                x_0=x_0,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = self.model(x_t, self._scale_timesteps(t), cond=cond, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
    
                    x_0=x_0,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_0=x_0, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_0,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_0.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms, {
            "pred": model_output,
            "x_t": x_t,
            "pred_x0": self._predict_x0_from_eps(x_t=x_t, t=t, eps=model_output),
            "target": noise,
        }

    def training_step(self, batch, batch_idx):
        imgs, metadata = batch 
        device = imgs.device
        self.imgs = imgs
        protein_ids = metadata["protein-id"]
        if self.condition_type == "class":
            cls_labels = [class_dict[key] for key in protein_ids]
            cond = torch.tensor(cls_labels, dtype=torch.int8).to(device).long()
        elif self.condition_type == "latent":
            latent_code = self.latent_encoder.forward_features(imgs) # Assuming latent encoder is a ViT and the global pooling method is set to either 'avg' or 'token'
            cond = latent_code
        else: 
            cond = None
        t = torch.randint(0, 1000, (imgs.shape[0],), device=device).long()
        losses, model_outputs = self(x_0=imgs, t=t, cond=cond)
        loss = losses["loss"].mean()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.9, 0.99))
        return [optimizer]
 
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t