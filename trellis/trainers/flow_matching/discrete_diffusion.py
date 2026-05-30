from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import os
import torch.distributed as dist
import matplotlib.pyplot as plt 
import io
from PIL import Image
import wandb
import re
from pytorch3d.ops import cubify
from trimesh import Trimesh
from einops import rearrange
from ..basic import BasicTrainer
from contextlib import nullcontext
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import DiscreteClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin
from ...utils.data_utils import recursive_to_device
from tqdm import tqdm
import json
from ...utils.mask_utils import generate_block_masks_ragged


# This script is build upon TRELLIS https://github.com/microsoft/TRELLIS and DUO https://github.com/s-sahoo/duo

"""
This is the trainer of uniform state discrete diffusion model (USDM).

"""

class LogLinear(torch.nn.Module):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

  def forward(self, t):
    t = (1 - self.eps) * t
    alpha_t = 1 - t 
    dalpha_t = - (1 - self.eps)
    return dalpha_t, alpha_t
  

class DiscreteDiffusionTrainer(BasicTrainer):
    """
    Trainer for diffusion model with USDM objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        vocab_size: int = 2,
        multi_mask_ratio: float = 0.0,
        use_cross_entropy: bool = False,
        reweight_elbo: bool=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.noise = LogLinear(sigma_min)
        self.vocab_size = vocab_size
        self.sigma_min = sigma_min
        self.multi_mask_ratio = multi_mask_ratio
        self.use_cross_entropy = use_cross_entropy  
        self.reweight_elbo = reweight_elbo

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None, multi_mask: Optional[bool]=False) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps. In other words, sample from q(x_t | x_0). 
        If multi_mask is True, the perturbation pattern will be blocked-structured (BSP). Actual_t is returned which calculates the proportion of noised tokens.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t: the noisy version of x_0 under timestep t, same shape as x_0.
            actual_t: Optional. Only returned when multi_mask is True. The actual t used for each sample, same shape as t.
        """
        if noise is None:
            noise = torch.randint(0, self.vocab_size, x_0.shape, device=x_0.device)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        if not multi_mask:
            _, alpha_t = self.noise(t)
            alpha_t = alpha_t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
            move_indices = torch.rand(x_0.shape, device=x_0.device) < 1 - alpha_t

            x_t = torch.where(move_indices, noise, x_0)
            return x_t
        else:
            # only works for 64
            lengths = torch.tensor([1,2,4,8,16,32,48,54,60], device = x_0.device)
            move_indices, actual_t = generate_block_masks_ragged(
                B=x_0.shape[0], N=64, dim=3, lengths=lengths, t=1-self.noise(t)[1], device=x_0.device, round_from=9
            )
            move_indices = move_indices.unsqueeze(1)


            # ------------------------- perturb t_new portion inside blocks ---------------------------
            t_new = self.sample_t(x_0.shape[0]).to(x_0.device).float()
            _, alpha_t = self.noise(t_new)
            alpha_t = alpha_t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
            move_indices_new = torch.rand(x_0.shape, device=x_0.device) < 1 - alpha_t
            move_indices_use = move_indices & move_indices_new 
            actual_t = t_new*actual_t

            x_t = torch.where(move_indices_use, noise, x_0)
            return x_t, actual_t



    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        Here assume noise is the sample of x_0 prediction.
        """
        # raise NotImplementedError
        assert noise.dtype in (torch.int32, torch.int64), "noise must be integer type"
        return noise

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        raise NotImplementedError # to be consistent with flow matching trainer

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.USDMAncestralSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.USDMAncestralSampler(self.sigma_min, self.vocab_size)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_beta_uniform_mix(self, n, p_beta=0.98):
        # vectorized mixture sampling: with probability p_beta draw from beta_dist, else uniform(0,1)
        # more t close to gt is sampled
        beta_dist =  torch.distributions.Beta(3,1)
        mask = torch.rand(n) < p_beta
        n_beta = int(mask.sum().item())
        out = torch.empty(n)
        if n_beta > 0:
            out[mask] = beta_dist.sample((n_beta,))
        if n - n_beta > 0:
            out[~mask] = torch.rand(n - n_beta)
        return out
    
    def sample_logitnormal_uniform_mix(self, n, p_logitnormal=0.90, mean=1.0, std=1.0):
        # vectorized mixture sampling: with probability p_logitnormal draw from logitnormal, else uniform(0,1)
        
        mask = torch.rand(n) < p_logitnormal
        n_lognormal = int(mask.sum().item())
        out = torch.empty(n)
        if n_lognormal > 0:
            out[mask] = torch.sigmoid(torch.randn(n_lognormal) * std + mean)
        if n - n_lognormal > 0:
            out[~mask] = torch.rand(n - n_lognormal)
        return out


    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """

        if self.t_schedule['name'] == 'uniform':
            # do antithetic sampling for t
            _eps_t = torch.rand(batch_size)
            offset = torch.arange(batch_size) / batch_size
            _eps_t = (_eps_t + offset) % 1
            t = _eps_t
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        elif self.t_schedule['name'] == 'beta_uniform':
            t = self.sample_beta_uniform_mix(batch_size, self.t_schedule['args']['p_beta'])
        elif self.t_schedule['name'] == 'logitNormal_uniform':
            t = self.sample_logitnormal_uniform_mix(batch_size, self.t_schedule['args']['p_logitnormal'], self.t_schedule['args']['mean'], self.t_schedule['args']['std'])
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t,):

        """
        Calculate the (variant of) negative log likelihood per token, used as diffusion loss. Adapted from https://github.com/s-sahoo/duo.
        
        Args:
            log_x_theta: The [N, L, vocab_size] tensor of log probabilities predicted by the model.
            xt: [N, L] int tensor of noised inputs at time t.
            x0:  [N, L] int tensor of original inputs.
            alpha_t: [N,1] tensor of alpha_t.
            d_alpha_t: scalar or [N,1] tensor of the time derivative of alpha_t.

        """

        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        assert xt.ndim == 2
        assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
        if not self.use_cross_entropy:
            if self.reweight_elbo:
                # reweight elbo to avoid large variance components in loss
                alpha_t = torch.clamp(alpha_t, max=0.6)

            x_reconst = log_x_theta.exp()
            x_bar_theta = self.vocab_size * alpha_t[
                :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
            coeff = dalpha_t / (self.vocab_size * alpha_t)
            x_eq_xt = (x0 == xt).float()
            x_neq_xt = 1 - x_eq_xt
            xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
            xbar_theta_xt = torch.gather(
            x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
            xbar_theta_x = torch.gather(
            x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
            term1 = self.vocab_size * (1 / xbar_xt
                                        - 1 / xbar_theta_xt)
            
            const = (1 - alpha_t) / (self.vocab_size * alpha_t
                                    + 1 - alpha_t)
            term2_coefs = x_eq_xt * const + x_neq_xt
            term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                            - (1 / const) * x_neq_xt) * const.log()
            term2_theta = - term2_coefs * (
            x_bar_theta.log().sum(-1)
            - self.vocab_size * xbar_theta_xt.log())
            term2_theta = (
            term2_theta
            - self.vocab_size * alpha_t / (1 - alpha_t) * (
                xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
            term2 = term2_theta + term2_offset
            diffusion_loss = coeff * (term1 - term2)
            assert diffusion_loss.ndim == 2
        else:
            # use cross entropy loss
            diffusion_loss = F.cross_entropy(log_x_theta.permute(0,2,1), x0, reduction='none')
        return diffusion_loss   

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        bsz = x_0.shape[0]
        noise = torch.randint(0, self.vocab_size, x_0.shape, device=x_0.device)
        if self.multi_mask_ratio > 0.0:
            # for multi_mask, we use t as uniform by default
            num_multi_mask = (torch.rand(bsz, device=x_0.device) < self.multi_mask_ratio).sum()
            if num_multi_mask > 0:
                t_multi = torch.rand(num_multi_mask, device=x_0.device)
                x_t_multi_mask, t_multi_actual = self.diffuse(x_0[:num_multi_mask], t_multi, noise[:num_multi_mask], multi_mask = True)
                t_origin_mask = self.sample_t(bsz - num_multi_mask).to(x_0.device).float()
                x_t_origin = self.diffuse(x_0[num_multi_mask:], t_origin_mask, noise[num_multi_mask:], multi_mask=False)

                x_t = torch.cat([x_t_multi_mask, x_t_origin], dim=0)
                t = torch.cat([t_multi_actual, t_origin_mask], dim=0)
            else:
                t = self.sample_t(bsz).to(x_0.device).float()
                x_t = self.diffuse(x_0, t, noise=noise)
        else:
            t = self.sample_t(bsz).to(x_0.device).float()
            x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred_logits = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        pred_logits_normalized = F.log_softmax(pred_logits, dim=-1) # shape [B,...,vocab_size]

        
        dalpha_t, alpha_t = self.noise(t)
        dalpha_t = torch.ones_like(alpha_t) * dalpha_t  
        # reshape to (B, L) while computing loss

        loss_nll = self.nll_per_token(
            rearrange(pred_logits_normalized, 'b d h w v -> b (d h w) v'),
            rearrange(x_t, 'b 1 d h w -> b (d h w)'),
            rearrange(x_0, 'b 1 d h w -> b (d h w)'),
            alpha_t.unsqueeze(-1),
            dalpha_t.unsqueeze(-1),
        )   
        terms = edict()
        terms["nll"] = loss_nll.mean()
        terms["loss"] = terms["nll"]

        # log loss with time bins, be consistent with continous
        with torch.no_grad():
            nll_per_instance = loss_nll.detach().mean(dim=-1)
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"nll": nll_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler() # overwrite by classifier_free_guidance mixin
        sample_gt = []
        sample = []
        uids: List[str] = []
        cond_vis = []
        # persistent iterator to avoid repeating the first batch every time
        data_iter = iter(dataloader)
        # choose autocast context matching training precision
        if self.fp16_mode == 'amp':
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        elif self.fp16_mode == 'bf16':
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(data_iter)
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            # collect uids if present
            if 'uid' in data:
                # ensure this is a list of strings
                batch_uids = data['uid']
                if isinstance(batch_uids, (list, tuple)):
                    uids.extend([str(x) for x in batch_uids])
                else:
                    # fallback to string, one per item expected
                    uids.extend([str(batch_uids)] * batch)

            noise = torch.randint(0, self.vocab_size, data['x_0'].shape, device=data['x_0'].device)
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            # ensure inference runs under autocast so FlashAttention sees half/bfloat tensors
            with autocast_ctx:
                res = sampler.sample(
                    self.models['denoiser'],
                    noise=noise,
                    **args,
                    steps=256, cfg_strength=2.0, verbose=verbose, rescale_t='cosine',
                ) # cfg here not used
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        assert sample.ndim == 5, "data should be [B, 1, D, H, W]"
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample_gt'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        # check in cond_vis is text or image

        if cond_vis[0]['cond_vis']['type'] == 'image':
            sample_dict.update(dict_reduce(cond_vis, None, {
                'value': lambda x: torch.cat(x, dim=0),
                'type': lambda x: x[0],
            }))
        elif cond_vis[0]['cond_vis']['type'] == 'text':
            sample_dict.update(dict_reduce(cond_vis, None, {
                'value': lambda x: x,
                'type': lambda x: x[0],
            }))
        if len(uids) > 0:
            sample_dict['uids'] = {'value': uids, 'type': 'uid_list'}
        
        return sample_dict


    @torch.no_grad()
    def slice_voxel(self, voxels, index = -1):
        """
        Slice voxels for visualization.
        Args:
            voxels: (B, D, H, W) tensor
            index: index to slice
        """
        pils = []
        assert voxels.ndim == 5, "voxels should be [B, 1, D, H, W]"
        voxels = voxels.squeeze(1)
        if index == -1:
            index = voxels.shape[-1] // 2 - 1 
        if voxels.shape[-1] == 1:
            num_samples = voxels.shape[0]
            fig, axes = plt.subplots(1, min(20,num_samples), figsize=(2*num_samples, 3))
            for i in range(min(20,num_samples)):
                img = voxels[i].reshape(28, 28).cpu().numpy()
                axes[i].imshow(img, cmap='gray', vmin=0, vmax=self.vocab_size-1)
                axes[i].axis('off')
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)              # important to free memory
            buf.seek(0)
            pils.append(Image.open(buf).convert("RGB"))
        else:
            for i in range(voxels.shape[0]):
                slice_x = voxels[i, :, index, :].cpu().numpy()
                slice_y = voxels[i, :, :, index].cpu().numpy()
                slice_z = voxels[i, index, :, :].cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(slice_x, cmap='gray')
                axs[0].set_title('Slice along X-axis')
                axs[0].axis('off')
                axs[1].imshow(slice_y, cmap='gray')
                axs[1].set_title('Slice along Y-axis')
                axs[1].axis('off')
                axs[2].imshow(slice_z, cmap='gray')
                axs[2].set_title('Slice along Z-axis')
                axs[2].axis('off')

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)              # important to free memory
                buf.seek(0)
                pils.append(Image.open(buf).convert("RGB"))
        return pils


    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=4, batch_size=4, verbose=False):
        """
        Sample images from the model.
        """
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose) 


        # Gather results
        if self.world_size > 1:
            # iterate over a copy of keys as we'll mutate values
            for key in list(samples.keys()):
                # Special handling for uid lists (non-tensor)
                if samples[key]['type'] == 'uid_list':
                    if self.is_master:
                        all_uids: Optional[List[List[str]]] = [None for _ in range(self.world_size)]
                    else:
                        all_uids = None
                    dist.gather_object(samples[key]['value'], all_uids, dst=0)
                    if self.is_master:
                        # flatten and trim
                        flat = []
                        for part in all_uids:
                            if part is not None:
                                flat.extend(part)
                        samples[key]['value'] = flat[:num_samples]
                    continue

                if samples[key]['type'] == 'text':
                    if self.is_master:
                        all_uids: Optional[List[List[str]]] = [None for _ in range(self.world_size)]
                    else:
                        all_uids = None
                    dist.gather_object(samples[key]['value'], all_uids, dst=0)
                    if self.is_master:
                        # flatten and trim
                        flat = []
                        for part in all_uids:
                            if part is not None:
                                flat.extend(part)
                        samples[key]['value'] = flat[:num_samples]
                    continue

                samples[key]['value'] = samples[key]['value'].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = None
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images
        if self.is_master:
            # helper to sanitize uid for filenames
            def _sanitize_uid(s: str) -> str:
                s = str(s)
                s = s.strip()
                # replace all non safe filename chars with '_'
                return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:128]

            # extracted uids if available
            uid_list: Optional[List[str]] = None
            if 'uids' in samples and isinstance(samples['uids'].get('value', None), list):
                uid_list = samples['uids']['value']

            text_list: Optional[List[str]] = None

            if samples['cond_vis']['type'] == 'text':
                text_list = samples['cond_vis']['value']

            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'gt', suffix), exist_ok=True)
            # dump all uids to json under samples/<suffix>/uids.json
            if uid_list is not None:
                uid_json_path = os.path.join(self.output_dir, 'samples', suffix, 'uids.json')
                try:
                    with open(uid_json_path, 'w', encoding='utf-8') as f:
                        import json as _json
                        _json.dump(uid_list, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[snapshot] Failed to write uids.json: {e}")

            if text_list is not None:
                text_json_path = os.path.join(self.output_dir, 'samples', suffix, 'text.json')
                try:
                    with open(text_json_path, 'w', encoding='utf-8') as f:
                        import json as _json
                        _json.dump(text_list, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[snapshot] Failed to write uids.json: {e}")
                    
            for key in samples.keys():

                if samples[key]['type'] == 'image':
                    wandb.log({'samples/condition': [wandb.Image(f) for f in samples[key]['value']]}, step=self.step)
                elif samples[key]['type'] == 'sample_gt':
                    # assume samples
                    # save voxel slices
                    pils = self.slice_voxel(samples[key]['value'])

                    wandb.log({'samples/voxel_slices_gt': [wandb.Image(f) for f in pils]}, step=self.step)
                    # log voxels
                    if samples[key]['value'].shape[-1] == 1:
                        continue
                    meshes_gt = []
                    cubified_mesh = cubify(samples[key]['value'].squeeze(1).float(), thresh=0.5, align="center")
                    for i, (verts, faces) in enumerate(zip(cubified_mesh.verts_list(), cubified_mesh.faces_list())):
                        uid_tag = _sanitize_uid(uid_list[i]) if uid_list is not None and i < len(uid_list) else str(i)
                        verts_np = verts.cpu().numpy()
                        faces_np = faces.cpu().numpy()
                        # export to obj
                        mesh = Trimesh(vertices=verts_np, faces=faces_np, process=False)
                        path = os.path.join(self.output_dir, 'gt', suffix, f'{key}_{suffix}_{uid_tag}.obj')
                        mesh.export(path)
                        meshes_gt.append(path)
                elif samples[key]['type'] == 'sample':
                    # assume samples
                    # save voxel slices
                    pils = self.slice_voxel(samples[key]['value'])
                    for i, pil in enumerate(pils):
                        uid_tag = _sanitize_uid(uid_list[i]) if uid_list is not None and i < len(uid_list) else str(i)
                        pil.save(os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}_{uid_tag}.png'))
                    
                    wandb.log({'samples/voxel_slices_generated': [wandb.Image(f) for f in pils]}, step=self.step)
                    # log voxels
                    if samples[key]['value'].shape[-1] == 1:
                        continue
                    meshes_sample = []
                    cubified_mesh = cubify(samples[key]['value'].squeeze(1).float(), thresh=0.5, align="center")
                    for i, (verts, faces) in enumerate(zip(cubified_mesh.verts_list(), cubified_mesh.faces_list())):
                        uid_tag = _sanitize_uid(uid_list[i]) if uid_list is not None and i < len(uid_list) else str(i)
                        verts_np = verts.cpu().numpy()
                        faces_np = faces.cpu().numpy()
                        # export to obj
                        mesh = Trimesh(vertices=verts_np, faces=faces_np, process=False)
                        path = os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}_{uid_tag}.obj')
                        mesh.export(path)
                        meshes_sample.append(path)
                        # save the corresponding voxels as npz
                        voxel_path = os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}_{uid_tag}.npz')
                        np.savez_compressed(voxel_path, voxels=samples[key]['value'][i].cpu().numpy())

                    
 

        if self.is_master:
            print(' Done.')
    

    def validation_step(self, data_list):
        """
        Run a training step.
        """
        
        # Choose autocast context: fp16 (amp), bf16, or disabled
        if self.fp16_mode == 'amp':
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        elif self.fp16_mode == 'bf16':
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()


        for i, mb_data in enumerate(data_list):
            
            # data_list should have length 1, need to set batch_split=1 in trainer
            with autocast_ctx:
                loss, status = self.eval_nll(**mb_data)
                l = loss / len(data_list)

        dicts = {'nll_sum': l}
        dicts.update(status)
        return dicts

    def eval_nll(
            self,
            x_0: torch.Tensor,
            cond=None,
            **kwargs
        ) -> Tuple[Dict, Dict]:
            """
            Compute training losses for a single timestep.

            Args:
                x_0: The [N x C x ...] tensor of noiseless inputs.
                cond: The [N x ...] tensor of additional conditions.
                kwargs: Additional arguments to pass to the backbone.

            Returns:
                a dict with the key "loss" containing a tensor of shape [N].
                may also contain other keys for different terms.
            """
            bsz = x_0.shape[0]
            noise = torch.randint(0, self.vocab_size, x_0.shape, device=x_0.device)
            t = self.sample_t(bsz).to(x_0.device).float()
            x_t = self.diffuse(x_0, t, noise=noise)
            cond = self.get_cond(cond, **kwargs)
            
            pred_logits = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
            pred_logits_normalized = F.log_softmax(pred_logits, dim=-1) # shape [B,...,vocab_size]
            
            dalpha_t, alpha_t = self.noise(t)
            dalpha_t = torch.ones_like(alpha_t) * dalpha_t  
            # reshape to (B, L) while computing loss
            log_x_theta = rearrange(pred_logits_normalized, 'b d h w v -> b (d h w) v')
            x0 = rearrange(x_0, 'b 1 d h w -> b (d h w)')
            log_p_theta = torch.gather(
                input=log_x_theta,
                dim=-1,
                index = x0[:,:, None]
            ).squeeze(-1)

            loss_nll =  -log_p_theta.mean(-1).sum()
            terms = edict()

            with torch.no_grad():
                nll_per_instance = -log_p_theta.detach().mean(dim=-1)
            time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
            for i in range(10):
                if (time_bin == i).sum() != 0:
                    terms[f"bin_{i}"] = {"nll": nll_per_instance[time_bin == i].cpu()}

            return loss_nll, terms


    @torch.no_grad()
    def validate_nll(self, runs: int = 1):
        if not self.is_master:
            return
        self.training_models['denoiser'].eval()
        avg_nll = torch.tensor(0.0, dtype = torch.float64)
        avg_nlls = []
        bin_collection = {f'bin_{i}': torch.tensor([]) for i in range(10)}

        # prepare dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )

        for run in range(runs):

            total_nll = torch.tensor(0.0, dtype = torch.float64)

            for this_data in tqdm(dataloader):
                if self.prefetch_data:
                    if self._data_prefetched is None:
                        self._data_prefetched = recursive_to_device(this_data, self.device, non_blocking=True)
                    data = self._data_prefetched
                    self._data_prefetched = recursive_to_device(this_data, self.device, non_blocking=True)
                else:
                    data = recursive_to_device(this_data, self.device, non_blocking=True)
                
                # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
                assert self.batch_split == 1, "NLL validation currently only supports batch_split=1"
                if isinstance(data, dict):
                    if self.batch_split == 1:
                        data_list = [data]
                    else:
                        batch_size = list(data.values())[0].shape[0]
                        data_list = [
                            {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                            for i in range(self.batch_split)
                        ]
                elif isinstance(data, list):
                    data_list = data
                else:
                    raise ValueError('Data must be a dict or a list of dicts.')

                # now we have data_list for data
                step_log = self.validation_step(data_list)
                total_nll += step_log['nll_sum'].cpu().double()
                print('Current NLL:', step_log['nll_sum'].item()/self.batch_size_per_gpu)
                # collect nlls in bins
                for i in range(10):
                    bin_key = f'bin_{i}'
                    if bin_key in step_log:
                        bin_collection[bin_key] = torch.concat(
                            [bin_collection[bin_key], step_log[bin_key]['nll'].double()]
                        )
            avg_nlls.append((total_nll / len(self.dataset)).item())
            avg_nll += total_nll / len(self.dataset) / runs
        print(f'Validation NLL: {avg_nll.item()}')
        path = os.path.join(self.output_dir, 'nlls.json')

        # calculate stats for bin collection
        bin_log = {}
        for i in range(10):
            bin_key = f'bin_{i}'
            if bin_collection[bin_key].numel() > 0:
                bin_log[bin_key] = {
                    'mean': bin_collection[bin_key].mean().item(),
                    'std': bin_collection[bin_key].std().item(),
                    'count': bin_collection[bin_key].numel(),
                }
            else:
                bin_log[bin_key] = {
                    'mean': None,
                    'std': None,
                    'count': 0,
                }
        print('NLL per time bin:', bin_log)
        logs = {}
        logs.update({
            'nlls': avg_nlls,
            'mean': avg_nll.item(),
            'std': np.std(avg_nlls),
            'bin_stats': bin_log
        })
        with open(path, 'w') as f:
            json.dump(logs, f, indent=4)

        # plot figrue and save
        import matplotlib.pyplot as plt
        bins = sorted(bin_log.keys(), key=lambda k: int(k.split('_')[-1]))
        means = [bin_log[k]['mean'] for k in bins]
        stds  = [bin_log[k]['std']  for k in bins]
        counts = [bin_log[k]['count'] for k in bins]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = range(len(bins))
        bars = ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x, bins, rotation=45, ha='right')
        ax.set_ylabel('Mean')
        ax.set_title('Mean ± Std per Bin')
        fig.tight_layout()
        out_path = 'bin_mean_std.png'
        fig.savefig(os.path.join(self.output_dir, out_path), dpi=300)
        plt.close(fig)

class DiscreteDiffusionCFGTrainer(DiscreteClassifierFreeGuidanceMixin, DiscreteDiffusionTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedDiscreteDiffusionCFGTrainer(TextConditionedMixin, DiscreteDiffusionCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedDiscreteDiffusionCFGTrainer(ImageConditionedMixin, DiscreteDiffusionCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass
