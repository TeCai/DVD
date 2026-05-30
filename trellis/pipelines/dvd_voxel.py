from dataclasses import dataclass
from typing import Callable, Literal

import torch

from .trellis_image_to_3d import TrellisImageTo3DPipeline
from .trellis_text_to_3d import TrellisTextTo3DPipeline
from .samplers import USDMAncestralGuidanceFunctionSampler

CfgSchedule = float | Callable[[float], float]

DVD_PRETRAINED_FILENAMES = {
    "image": {
        "base": ("dvd_img.json", "dvd_img.safetensors"),
        "bsp": ("dvd_img_BSP_ft.json", "dvd_img_BSP_ft.safetensors"),
    },
    "text": {
        "base": ("dvd_text.json", "dvd_text.safetensors"),
        "bsp": ("dvd_text_BSP_ft.json", "dvd_text_BSP_ft.safetensors"),
    },
}
DVD_VARIANT_ALIASES = {
    "default": "base",
    "generation": "base",
    "gen": "base",
    "edit": "bsp",
    "editing": "bsp",
    "bsp_ft": "bsp",
    "bsp-ft": "bsp",
    "bsp_finetune": "bsp",
    "bsp-finetune": "bsp",
}


def _dvd_pretrained_filenames(
    modality: Literal["image", "text"],
    variant: str = "base",
    config_name: str | None = None,
    checkpoint_name: str | None = None,
) -> tuple[str, str]:
    if (config_name is None) != (checkpoint_name is None):
        raise ValueError("Pass both config_name and checkpoint_name, or neither.")
    if config_name is not None and checkpoint_name is not None:
        return config_name, checkpoint_name

    variant_key = DVD_VARIANT_ALIASES.get(variant.lower(), variant.lower())
    try:
        return DVD_PRETRAINED_FILENAMES[modality][variant_key]
    except KeyError as exc:
        valid = ", ".join(sorted(DVD_PRETRAINED_FILENAMES[modality]))
        raise ValueError(f"Unknown DVD {modality} variant '{variant}'. Valid variants: {valid}.") from exc


def _resolve_dvd_pretrained_files(
    pretrained_model_name_or_path: str,
    config_name: str,
    checkpoint_name: str,
    subfolder: str | None = None,
    revision: str | None = None,
    repo_type: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    token: bool | str | None = None,
) -> tuple[str, str]:
    from pathlib import Path

    local_root = Path(pretrained_model_name_or_path)
    if local_root.is_dir():
        local_root = local_root / subfolder if subfolder else local_root
        config_path = local_root / config_name
        checkpoint_path = local_root / checkpoint_name
        if not config_path.exists():
            raise FileNotFoundError(f"DVD config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DVD checkpoint not found: {checkpoint_path}")
        return str(config_path), str(checkpoint_path)

    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": pretrained_model_name_or_path,
        "subfolder": subfolder,
        "revision": revision,
        "repo_type": repo_type,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "token": token,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    config_path = hf_hub_download(filename=config_name, **kwargs)
    checkpoint_path = hf_hub_download(filename=checkpoint_name, **kwargs)
    return config_path, checkpoint_path


@dataclass
class DVDVoxelOutput:
    """
    DVD voxel samples in DVD convention.

    Attributes:
        samples: Occupancy tensor with shape [B, D, H, W].
        coords: Occupied coordinates with shape [N, 4] in [batch, x, y, z]
            DVD convention.
    """
    samples: torch.Tensor
    coords: torch.Tensor

    @property
    def coords_without_batch(self) -> torch.Tensor:
        return self.coords[:, 1:]

    def to_trellis_coords(self, resolution: int = 64, device: torch.device | str | None = None) -> torch.Tensor:
        coords = dvd_coords_to_trellis_coords(self.coords, resolution=resolution)
        if device is not None:
            coords = coords.to(device)
        return coords


def load_dvd_denoiser(config_path: str, checkpoint_path: str, device: torch.device | str | None = None) -> torch.nn.Module:
    import json
    from safetensors.torch import load_file
    from trellis import models

    with open(config_path, "r") as f:
        config = json.load(f)

    denoiser = getattr(models, config["name"])(**config["args"])
    denoiser.load_state_dict(load_file(checkpoint_path))
    denoiser.eval()
    if device is not None:
        denoiser.to(device)
    return denoiser


def image_cfg_schedule(t: float) -> float:
    return 0.4 if t < 0.5 else 0.7


def text_cfg_schedule(t: float) -> float:
    return 0.4 if t < 0.8 else 1.2


def bsp_edit_cfg_schedule(t: float) -> float:
    return 0.45


def normalize_cfg_schedule(cfg_strength: CfgSchedule):
    if callable(cfg_strength):
        return cfg_strength
    return lambda _t: float(cfg_strength)


def coords_to_samples(coords: torch.Tensor, resolution: int = 64, batch_size: int | None = None) -> torch.Tensor:
    coords = coords.detach().cpu().long()
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError("coords must have shape [N, 3] or [N, 4]")

    if coords.shape[1] == 3:
        coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=coords.dtype), coords], dim=1)

    if batch_size is None:
        batch_size = int(coords[:, 0].max().item()) + 1 if coords.numel() > 0 else 1

    samples = torch.zeros((batch_size, resolution, resolution, resolution), dtype=torch.long)
    samples[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 1
    return samples


def samples_to_coords(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim == 5 and samples.shape[1] == 1:
        samples = samples.squeeze(1)
    if samples.ndim != 4:
        raise ValueError("samples must have shape [B, D, H, W] or [B, 1, D, H, W]")
    return torch.argwhere(samples.detach().cpu() > 0).int()


def as_voxel_output(voxels: DVDVoxelOutput | torch.Tensor, resolution: int = 64) -> DVDVoxelOutput:
    if isinstance(voxels, DVDVoxelOutput):
        return voxels
    if not isinstance(voxels, torch.Tensor):
        raise TypeError("voxels must be DVDVoxelOutput or torch.Tensor")

    if voxels.ndim in (4, 5):
        samples = voxels.detach().cpu()
        if samples.ndim == 5 and samples.shape[1] == 1:
            samples = samples.squeeze(1)
        return DVDVoxelOutput(samples=samples.long(), coords=samples_to_coords(samples))

    if voxels.ndim == 2 and voxels.shape[1] in (3, 4):
        coords = voxels.detach().cpu().int()
        if coords.shape[1] == 3:
            coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=coords.dtype), coords], dim=1)
        return DVDVoxelOutput(samples=coords_to_samples(coords, resolution=resolution), coords=coords)

    raise ValueError("voxel tensor must be samples [B, D, H, W] or coords [N, 3/4]")


def dvd_coords_to_trellis_coords(coords: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    coords = coords.detach().clone().int()
    has_batch = coords.shape[1] == 4
    if not has_batch:
        coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=coords.dtype), coords], dim=1)
    coords = coords[:, [0, 1, 3, 2]]
    coords[:, 1] = resolution - 1 - coords[:, 1].clone()
    return coords if has_batch else coords[:, 1:]


def trellis_coords_to_dvd_coords(coords: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    coords = coords.detach().clone().int()
    has_batch = coords.shape[1] == 4
    if not has_batch:
        coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=coords.dtype), coords], dim=1)
    coords = coords[:, [0, 1, 3, 2]]
    coords[:, 1] = resolution - 1 - coords[:, 1].clone()
    return coords if has_batch else coords[:, 1:]


def make_box_mask(
    resolution: int = 64,
    batch_size: int = 1,
    x: tuple[int, int] | None = None,
    y: tuple[int, int] | None = None,
    z: tuple[int, int] | None = None,
    mode: Literal["keep", "edit"] = "keep",
) -> torch.Tensor:
    """
    Build a boolean mask with shape [B, D, H, W].

    The sampler expects True for known voxels to keep. Use mode="edit" when
    describing the region to regenerate; the returned mask is inverted for you.
    """
    edit_mask = torch.zeros((batch_size, resolution, resolution, resolution), dtype=torch.bool)
    xs = x or (0, resolution)
    ys = y or (0, resolution)
    zs = z or (0, resolution)
    edit_mask[:, xs[0]:xs[1], ys[0]:ys[1], zs[0]:zs[1]] = True
    return ~edit_mask if mode == "edit" else edit_mask


def export_cubified_voxels(
    voxels: DVDVoxelOutput | torch.Tensor,
    output_path: str,
    voxel_size: float = 0.5,
    resolution: int = 64,
    color: tuple[int, int, int, int] = (150, 150, 150, 255),
) -> str:
    import trimesh
    from pytorch3d.ops import cubify

    output = as_voxel_output(voxels, resolution=resolution)
    samples = output.samples.float()
    cubified_meshes = cubify(samples.flip(dims=[-3]), voxel_size, align="center")
    vertices = cubified_meshes.verts_packed().cpu().numpy()
    faces = cubified_meshes.faces_packed().cpu().numpy()
    if vertices.size:
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        center = (vmin + vmax) / 2.0
        scale = (vmax - vmin).max() / 2.0
        if scale > 0:
            vertices = (vertices - center) / scale
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
    )
    mesh.visual.face_colors = color
    mesh.export(output_path)
    return output_path


class DVDVoxelPipelineMixin:
    resolution: int
    dvd_sampler: USDMAncestralGuidanceFunctionSampler

    @property
    def denoiser(self) -> torch.nn.Module:
        return self.models["denoiser"]

    def _init_dvd_sampler(self, resolution: int = 64, vocab_size: int = 2):
        self.resolution = resolution
        self.dvd_sampler = USDMAncestralGuidanceFunctionSampler(sigma_min=1e-5, vocab_size=vocab_size)

    def _sample_from_cond(
        self,
        cond: dict[str, torch.Tensor],
        num_samples: int = 1,
        steps: int = 256,
        cfg_strength: CfgSchedule = image_cfg_schedule,
        seed: int | None = None,
        rescale_t: str | float = "cosine",
        verbose: bool = True,
        noise: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
        classes: torch.Tensor | None = None,
    ) -> DVDVoxelOutput:
        if seed is not None:
            torch.manual_seed(seed)

        device = self.device
        cond = {k: v.to(device) for k, v in cond.items()}
        if noise is None:
            noise = torch.randint(
                0,
                2,
                (num_samples, 1, self.resolution, self.resolution, self.resolution),
                device=device,
            )
        else:
            noise = noise.to(device)

        sample_kwargs = {}
        if keep_mask is not None or classes is not None:
            if keep_mask is None or classes is None:
                raise ValueError("keep_mask and classes must be provided together")
            sample_kwargs["mask"] = keep_mask.to(device).bool()
            sample_kwargs["classes"] = classes.to(device).long()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            res = self.dvd_sampler.sample(
                self.denoiser,
                noise=noise,
                **cond,
                steps=steps,
                cfg_strength=normalize_cfg_schedule(cfg_strength),
                verbose=verbose,
                rescale_t=rescale_t,
                **sample_kwargs,
            )

        samples = res.samples.squeeze(1).detach().cpu().long()
        return DVDVoxelOutput(samples=samples, coords=samples_to_coords(samples))

    def edit_voxels_from_cond(
        self,
        cond: dict[str, torch.Tensor],
        voxels: DVDVoxelOutput | torch.Tensor,
        keep_mask: torch.Tensor,
        steps: int = 128,
        cfg_strength: CfgSchedule = bsp_edit_cfg_schedule,
        seed: int | None = None,
        rescale_t: str | float = "cosine",
        verbose: bool = True,
        init: Literal["base", "random"] = "base",
    ) -> DVDVoxelOutput:
        base = as_voxel_output(voxels, resolution=self.resolution).samples
        if keep_mask.shape != base.shape:
            raise ValueError(f"keep_mask must have shape {tuple(base.shape)}, got {tuple(keep_mask.shape)}")

        if seed is not None:
            torch.manual_seed(seed)

        if init == "base":
            init_samples = base.clone()
            init_samples[~keep_mask.cpu()] = torch.randint(0, 2, init_samples[~keep_mask.cpu()].shape)
        elif init == "random":
            init_samples = torch.randint(0, 2, base.shape)
        else:
            raise ValueError(f"Unsupported init mode: {init}")

        return self._sample_from_cond(
            cond,
            num_samples=base.shape[0],
            steps=steps,
            cfg_strength=cfg_strength,
            seed=None,
            rescale_t=rescale_t,
            verbose=verbose,
            noise=init_samples.unsqueeze(1),
            keep_mask=keep_mask,
            classes=base,
        )


class DVDImageToVoxelPipeline(DVDVoxelPipelineMixin, TrellisImageTo3DPipeline):
    def __init__(self, denoiser: torch.nn.Module, image_cond_model: str = "dinov2_vitl14_reg", resolution: int = 64):
        TrellisImageTo3DPipeline.__init__(self, models={"denoiser": denoiser}, image_cond_model=image_cond_model)
        self._init_dvd_sampler(resolution=resolution)

    def preprocess_image(self, image):
        """
        Preprocess image conditions for DVD voxel generation.

        This follows the TRELLIS foreground crop geometry, but pads and
        composites with a white background to match DVD's training setup.
        """
        import numpy as np
        import rembg
        from PIL import Image, ImageOps

        has_alpha = False
        if image.mode == "RGBA":
            alpha = np.array(image.getchannel("A"))
            if not np.all(alpha == 255):
                has_alpha = True

        if has_alpha:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
            max_size = max(image.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, "rembg_session", None) is None:
                self.rembg_session = rembg.new_session("u2net")
            image = rembg.remove(image, session=self.rembg_session).convert("RGBA")

        mask_np = np.array(image.getchannel("A"))
        ys, xs = np.nonzero(mask_np > 0.8*255)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("No foreground pixels found in image alpha channel.")

        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        hsize = max(x1 - x0, y1 - y0) / 2.0
        aug_hsize = hsize * 1.1

        ax0 = int(center_x - aug_hsize)
        ay0 = int(center_y - aug_hsize)
        ax1 = int(center_x + aug_hsize)
        ay1 = int(center_y + aug_hsize)

        width, height = image.size
        pad_l = max(0, -ax0)
        pad_t = max(0, -ay0)
        pad_r = max(0, ax1 - width)
        pad_b = max(0, ay1 - height)
        if pad_l or pad_t or pad_r or pad_b:
            image = ImageOps.expand(
                image,
                border=(pad_l, pad_t, pad_r, pad_b),
                fill=(255, 255, 255, 255),
            )
            ax0 += pad_l
            ay0 += pad_t
            ax1 += pad_l
            ay1 += pad_t
            width, height = image.size

        ax0 = max(0, ax0)
        ay0 = max(0, ay0)
        ax1 = min(width, ax1)
        ay1 = min(height, ay1)
        image = image.crop((ax0, ay0, ax1, ay1))
        image = image.resize((518, 518), Image.Resampling.LANCZOS)

        if image.mode == "RGBA":
            white = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(white, image).convert("RGB")
        else:
            image = image.convert("RGB")
        return image

    @classmethod
    def from_files(
        cls,
        config_path: str,
        checkpoint_path: str,
        image_cond_model: str = "dinov2_vitl14_reg",
        resolution: int = 64,
        device: torch.device | str | None = None,
    ) -> "DVDImageToVoxelPipeline":
        denoiser = load_dvd_denoiser(config_path, checkpoint_path)
        pipeline = cls(denoiser, image_cond_model=image_cond_model, resolution=resolution)
        if device is not None:
            pipeline.to(torch.device(device))
        return pipeline

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        variant: str = "base",
        config_name: str | None = None,
        checkpoint_name: str | None = None,
        image_cond_model: str = "dinov2_vitl14_reg",
        resolution: int = 64,
        device: torch.device | str | None = None,
        subfolder: str | None = None,
        revision: str | None = None,
        repo_type: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
    ) -> "DVDImageToVoxelPipeline":
        config_name, checkpoint_name = _dvd_pretrained_filenames("image", variant, config_name, checkpoint_name)
        config_path, checkpoint_path = _resolve_dvd_pretrained_files(
            pretrained_model_name_or_path,
            config_name,
            checkpoint_name,
            subfolder=subfolder,
            revision=revision,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        return cls.from_files(
            config_path,
            checkpoint_path,
            image_cond_model=image_cond_model,
            resolution=resolution,
            device=device,
        )

    @torch.no_grad()
    def sample_voxels(
        self,
        image,
        num_samples: int = 1,
        seed: int | None = None,
        steps: int = 256,
        cfg_strength: CfgSchedule = image_cfg_schedule,
        preprocess_image: bool = True,
        rescale_t: str | float = "cosine",
        verbose: bool = True,
    ) -> DVDVoxelOutput:
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        return self._sample_from_cond(cond, num_samples, steps, cfg_strength, seed, rescale_t, verbose)

    @torch.no_grad()
    def edit_voxels(
        self,
        image,
        voxels: DVDVoxelOutput | torch.Tensor,
        keep_mask: torch.Tensor,
        seed: int | None = None,
        steps: int = 128,
        cfg_strength: CfgSchedule = bsp_edit_cfg_schedule,
        preprocess_image: bool = True,
        rescale_t: str | float = "cosine",
        verbose: bool = True,
        init: Literal["base", "random"] = "base",
    ) -> DVDVoxelOutput:
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        return self.edit_voxels_from_cond(cond, voxels, keep_mask, steps, cfg_strength, seed, rescale_t, verbose, init)


class DVDTextToVoxelPipeline(DVDVoxelPipelineMixin, TrellisTextTo3DPipeline):
    def __init__(self, denoiser: torch.nn.Module, text_cond_model: str, resolution: int = 64):
        TrellisTextTo3DPipeline.__init__(self, models={"denoiser": denoiser}, text_cond_model=text_cond_model)
        self._init_dvd_sampler(resolution=resolution)

    @classmethod
    def from_files(
        cls,
        config_path: str,
        checkpoint_path: str,
        text_cond_model: str = "openai/clip-vit-large-patch14",
        resolution: int = 64,
        device: torch.device | str | None = None,
    ) -> "DVDTextToVoxelPipeline":
        denoiser = load_dvd_denoiser(config_path, checkpoint_path)
        pipeline = cls(denoiser, text_cond_model=text_cond_model, resolution=resolution)
        if device is not None:
            pipeline.to(torch.device(device))
        return pipeline

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        variant: str = "base",
        config_name: str | None = None,
        checkpoint_name: str | None = None,
        text_cond_model: str = "openai/clip-vit-large-patch14",
        resolution: int = 64,
        device: torch.device | str | None = None,
        subfolder: str | None = None,
        revision: str | None = None,
        repo_type: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
    ) -> "DVDTextToVoxelPipeline":
        config_name, checkpoint_name = _dvd_pretrained_filenames("text", variant, config_name, checkpoint_name)
        config_path, checkpoint_path = _resolve_dvd_pretrained_files(
            pretrained_model_name_or_path,
            config_name,
            checkpoint_name,
            subfolder=subfolder,
            revision=revision,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        return cls.from_files(
            config_path,
            checkpoint_path,
            text_cond_model=text_cond_model,
            resolution=resolution,
            device=device,
        )

    @torch.no_grad()
    def sample_voxels(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int | None = None,
        steps: int = 128,
        cfg_strength: CfgSchedule = text_cfg_schedule,
        rescale_t: str | float = "cosine",
        verbose: bool = True,
    ) -> DVDVoxelOutput:
        cond = self.get_cond([prompt])
        return self._sample_from_cond(cond, num_samples, steps, cfg_strength, seed, rescale_t, verbose)

    @torch.no_grad()
    def edit_voxels(
        self,
        prompt: str,
        voxels: DVDVoxelOutput | torch.Tensor,
        keep_mask: torch.Tensor,
        seed: int | None = None,
        steps: int = 128,
        cfg_strength: CfgSchedule = bsp_edit_cfg_schedule,
        rescale_t: str | float = "cosine",
        verbose: bool = True,
        init: Literal["base", "random"] = "base",
    ) -> DVDVoxelOutput:
        cond = self.get_cond([prompt])
        return self.edit_voxels_from_cond(cond, voxels, keep_mask, steps, cfg_strength, seed, rescale_t, verbose, init)


def run_image_stage2_from_dvd_voxels(
    trellis_pipeline: TrellisImageTo3DPipeline,
    image,
    voxels: DVDVoxelOutput | torch.Tensor,
    seed: int = 42,
    slat_sampler_params: dict | None = None,
    formats: list[str] | None = None,
    preprocess_image: bool = True,
    resolution: int = 64,
) -> dict:
    output = as_voxel_output(voxels, resolution=resolution)
    coords = output.to_trellis_coords(resolution=resolution, device=trellis_pipeline.device)
    return trellis_pipeline.run(
        image,
        seed=seed,
        slat_sampler_params=slat_sampler_params or {},
        formats=formats or ["mesh", "gaussian", "radiance_field"],
        preprocess_image=preprocess_image,
        coords=coords,
    )


def run_text_stage2_from_dvd_voxels(
    trellis_pipeline: TrellisTextTo3DPipeline,
    prompt: str,
    voxels: DVDVoxelOutput | torch.Tensor,
    seed: int = 42,
    slat_sampler_params: dict | None = None,
    formats: list[str] | None = None,
    resolution: int = 64,
) -> dict:
    output = as_voxel_output(voxels, resolution=resolution)
    coords = output.to_trellis_coords(resolution=resolution, device=trellis_pipeline.device)
    return trellis_pipeline.run(
        prompt=prompt,
        seed=seed,
        slat_sampler_params=slat_sampler_params or {},
        formats=formats or ["mesh", "gaussian", "radiance_field"],
        coords=coords,
    )
