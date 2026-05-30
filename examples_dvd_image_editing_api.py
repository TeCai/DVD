import argparse
import os
import torch

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

from PIL import Image

from trellis.pipelines import (
    DVDImageToVoxelPipeline,
    TrellisImageTo3DPipeline,
    as_voxel_output,
    export_cubified_voxels,
    run_image_stage2_from_dvd_voxels,
)
from trellis.utils import postprocessing_utils


def parse_args():
    parser = argparse.ArgumentParser(description="DVD voxel editing followed by TRELLIS stage 2.")
    parser.add_argument("--target-image", required=True, help="Target image condition for voxel editing.")
    parser.add_argument(
        "--voxel-coords",
        required=True,
        help="Existing voxel coords in DVD convention. Supports .npy, .pt, and .pth.",
    )
    parser.add_argument("--dvd-config", default="ckpts/dvd_img_BSP_ft.json", help="BSP fine-tuned DVD model config JSON.")
    parser.add_argument(
        "--dvd-checkpoint",
        default="ckpts/dvd_img_BSP_ft.safetensors",
        help="BSP fine-tuned DVD safetensors checkpoint.",
    )
    parser.add_argument("--output-dir", default="example_results", help="Directory for generated assets.")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution for loaded coords.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_voxel_coords(path):
    import numpy as np
    import torch

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.load(path)
    elif ext in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            for key in ("coords", "voxels", "samples"):
                if key in data:
                    data = data[key]
                    break
    else:
        raise ValueError(f"Unsupported voxel coord file extension: {ext}")
    return torch.as_tensor(data)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    target_image = Image.open(args.target_image)
    target_name = os.path.splitext(os.path.basename(args.target_image))[0]
    voxel_name = os.path.splitext(os.path.basename(args.voxel_coords))[0]
    name = f"{target_name}_{voxel_name}"

    dvd = DVDImageToVoxelPipeline.from_files(
        args.dvd_config,
        args.dvd_checkpoint,
        resolution=args.resolution,
        device=args.device,
    )
    voxels = as_voxel_output(load_voxel_coords(args.voxel_coords), resolution=args.resolution)

    # Loaded occupied coords, perturb the upper half.
    # Edit the place where keep_mask=0. To design your own editing mask, set keep_mask=1 for voxels you want to keep unchanged, and 0 for voxels you want to edit.

    keep_mask = torch.ones_like(voxels.samples)
    keep_mask[...,32:,:] = 0
    keep_mask = keep_mask.bool()

    edited_voxels = dvd.edit_voxels(target_image, voxels, keep_mask=keep_mask, seed=args.seed)
    export_cubified_voxels(edited_voxels, os.path.join(args.output_dir, f"{name}_edited_voxels.glb"))

    trellis = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    trellis.to(args.device)
    outputs = run_image_stage2_from_dvd_voxels(trellis, target_image, edited_voxels, seed=args.seed)

    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0])
    glb.export(os.path.join(args.output_dir, f"{name}_edited.glb"))


if __name__ == "__main__":
    main()
