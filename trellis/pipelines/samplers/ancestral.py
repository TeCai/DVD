from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import DiscreteClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import DiscreteGuidanceIntervalSamplerMixin
from .guidance_function_mixin import DiscreteGuidanceFunctionSamplerMixin
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical

class LogLinear(torch.nn.Module):
  def __init__(self, eps=1e-5):
    super().__init__()
    self.eps = eps  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

  def forward(self, t):
    t = (1 - self.eps) * t
    alpha_t = 1 - t 
    dalpha_t = - (1 - self.eps)
    return dalpha_t, alpha_t
  
def sample_categorical(categorical_probs):
  '''
  Sample from unormalized categorical distribution using Gumbel-max trick.
  '''
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)

class USDMAncestralSampler(Sampler):
    """
    Generate samples with ancestral sampling from a discrete diffusion model.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
        vocab_size: int,
    ):
        self.sigma_min = sigma_min
        self.noise = LogLinear(sigma_min)
        self.vocab_size = vocab_size
        self.mask=None
        self.classes=None

    def _eps_to_xstart(self, x_t, t, eps):
        raise NotImplementedError 

    def _xstart_to_eps(self, x_t, t, x_0):
        raise NotImplementedError 
    
    def _v_to_xstart_eps(self, x_t, t, v):
        raise NotImplementedError

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None:
            if isinstance(cond, torch.Tensor):
                if  cond.shape[0] == 1 and x_t.shape[0] > 1:
                    cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
            else:
                for k, v in cond.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1 and x_t.shape[0] > 1:
                        cond[k] = v.repeat(x_t.shape[0], *([1] * (len(v.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_logits = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0 = torch.argmax(pred_logits, dim=-1)
        return pred_x_0, pred_logits

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
        x: Either clean input `x0` (one-hot),
            or model's predicted `x_theta` of shape (B, L, V).
        xt: The noisy latent (as indices) of shape (B, L).
        alpha_s: Noise level at s of shape (B, [L | 1], 1).
        alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
        Posterior / approximate posterior of shape (B, L, V).
        """

        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(
        xt.dtype).to(xt.device)
        return (
        (alpha_t * self.vocab_size * x * xt_one_hot + (
            alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
            1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
                alpha_t * self.vocab_size * torch.gather(
                x, -1, xt[..., None]) + (1 - alpha_t)))

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        noise_removal: bool = False,
        temp : float = 1.0,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        # for this input x_t is of shape [B, 1, D, H, W]
        d,h,w = x_t.shape[-3:]
        pred_x_0, pred_logits = self._get_model_prediction(model, x_t, t, cond, **kwargs) #[b,d,h,w] [b,d,h,w,vocab]

        # make sure float64 when computing probs and sampling
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            pred_logits = pred_logits.to(torch.float64) * temp
        
            pred_prob = F.softmax(pred_logits, dim=-1)
            _, alpha_s = self.noise(t_prev)
            _, alpha_t = self.noise(t)

            # if mask is provided, inpaint the maksed regions to probability 1.0 on the given classes
            if self.mask is not None:
                one_hot = F.one_hot(self.classes, num_classes=self.vocab_size).to(pred_prob.dtype)
                pred2 = torch.where(self.mask.unsqueeze(-1), one_hot, pred_prob)
                pred_prob = pred2

            pred_prob = rearrange(pred_prob, 'b d h w v -> b (d h w) v')
            x_t = rearrange(x_t, 'b 1 d h w -> b (d h w)')
            # compute posterior q(x_s|x_t)
            
            if not noise_removal:
                # ancestral sampling
                x_prev_prob = self._compute_posterior(pred_prob, x_t, alpha_s, alpha_t)
                pred_x_prev = Categorical(probs=x_prev_prob).sample()
                pred_x_prev = sample_categorical(x_prev_prob)
            else:
                # greedy noise removal, according to https://arxiv.org/abs/2506.10892
                pred_x_prev = torch.argmax(pred_prob, dim=-1)
            


        # rearrange back to voxel shape of shape [B,1,D,H,W]
        pred_x_prev = rearrange(pred_x_prev, 'b (d h w) -> b 1 d h w', d=d, h=h, w=w)

        # replace places where mask==1 to classes
        if self.mask is not None:
            pred_x_prev = torch.where(self.mask.unsqueeze(1), self.classes.unsqueeze(1), pred_x_prev)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0.unsqueeze(1)})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        classes: torch.Tensor=None,
        mask: torch.Tensor=None,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method. For generating from scratch, leave classes and mask as None. For
        inpainting, provide both classes and mask.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            classes: Optional class (known) information for inpainting, shape [B, D, H, W], with gt indices.
            mask: Optional mask for inpainting, shape [B, D, H, W], with 1 regioins should use class info given by classes
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """

        self.model = model
        # classes and mask should be provided simultaneously
        if (classes is None) != (mask is None):
            raise ValueError('classes and mask should be provided simultaneously')

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        if isinstance(rescale_t, float) :
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        elif rescale_t == 'cosine':
            t_seq = np.cos(np.pi*(1-t_seq)/2)
        elif rescale_t == 'cosine_full':
            t_seq = np.cos(np.pi*(1-t_seq))/2 + 0.5
        else:
            raise NotImplementedError
        
        if mask is not None:
            self.create_mask(mask, classes)

        
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Sampling", disable=not verbose)):
            noise_removal = (i == len(t_pairs) - 1)
            out = self.sample_once(model, sample, t, t_prev, cond, noise_removal=noise_removal, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        
        if mask is not None:
            self.flush_mask()
        return ret

    def create_mask(self, mask: torch.Tensor, classes: torch.Tensor):
        self.mask = mask.to(self.model.device)
        self.classes = classes.to(self.model.device)

    def flush_mask(self,):
        self.mask = None
        self.classes = None

class USDMAncestralCfgSampler(DiscreteClassifierFreeGuidanceSamplerMixin, USDMAncestralSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class USDMAncestralGuidanceIntervalSampler(DiscreteGuidanceIntervalSamplerMixin, USDMAncestralSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)


class USDMAncestralGuidanceFunctionSampler(DiscreteGuidanceFunctionSamplerMixin, USDMAncestralSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: Callable[[float], float] = lambda t: 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength function of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)