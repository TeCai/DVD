import argparse
import os
import re

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from trellis.pipelines import (
    DVDTextToVoxelPipeline,
    TrellisTextTo3DPipeline,
    as_voxel_output,
    export_cubified_voxels,
    run_text_stage2_from_dvd_voxels,
)
from trellis.utils import postprocessing_utils


def slugify(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (slug[:max_length].strip("_") or "text_prompt")


def parse_range(values: list[int], resolution: int) -> tuple[int, int]:
    lo, hi = values
    lo = max(0, min(resolution, int(lo)))
    hi = max(0, min(resolution, int(hi)))
    if lo >= hi:
        raise ValueError(f"Invalid edit range {values}; expected start < end after clamping.")
    return lo, hi


def load_voxel_coords(path):
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


def make_keep_mask(voxels, edit_x: tuple[int, int], edit_y: tuple[int, int], edit_z: tuple[int, int]):
    keep_mask = torch.ones_like(voxels.samples, dtype=torch.bool)
    keep_mask[:, edit_x[0]:edit_x[1], edit_y[0]:edit_y[1], edit_z[0]:edit_z[1]] = False
    return keep_mask


def parse_args():
    parser = argparse.ArgumentParser(description="DVD text-conditioned voxel editing followed by TRELLIS stage 2.")
    parser.add_argument("--prompt", required=True, help="Target text condition for voxel editing.")
    parser.add_argument(
        "--voxel-coords",
        required=True,
        help="Existing voxel coords in DVD convention. Supports .npy, .pt, and .pth.",
    )
    parser.add_argument("--name", default=None, help="Output asset name. Defaults to a slug of the prompt and voxel file.")
    parser.add_argument("--dvd-config", default="ckpts/dvd_text_BSP_ft.json", help="BSP fine-tuned DVD text config JSON.")
    parser.add_argument(
        "--dvd-checkpoint",
        default="ckpts/dvd_text_BSP_ft.safetensors",
        help="BSP fine-tuned DVD text safetensors checkpoint.",
    )
    parser.add_argument("--trellis-model", default="microsoft/TRELLIS-text-large", help="TRELLIS text pipeline.")
    parser.add_argument("--output-dir", default="example_results", help="Directory for generated assets.")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution for loaded coords.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dvd-steps", type=int, default=128)
    parser.add_argument("--stage2-steps", type=int, default=25)
    parser.add_argument("--stage2-cfg", type=float, default=5.0)
    parser.add_argument("--edit-x", type=int, nargs=2, default=(0, 64), metavar=("START", "END"))
    parser.add_argument("--edit-y", type=int, nargs=2, default=(0, 64), metavar=("START", "END"))
    parser.add_argument("--edit-z", type=int, nargs=2, default=(32, 64), metavar=("START", "END"))
    parser.add_argument("--skip-stage2", action="store_true", help="Only generate and save edited DVD voxels.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    voxel_name = os.path.splitext(os.path.basename(args.voxel_coords))[0]
    name = args.name or f"{slugify(args.prompt)}_{voxel_name}"

    dvd = DVDTextToVoxelPipeline.from_files(
        args.dvd_config,
        args.dvd_checkpoint,
        resolution=args.resolution,
        device=args.device,
    )
    voxels = as_voxel_output(load_voxel_coords(args.voxel_coords), resolution=args.resolution)

    edit_x = parse_range(args.edit_x, args.resolution)
    edit_y = parse_range(args.edit_y, args.resolution)
    edit_z = parse_range(args.edit_z, args.resolution)
    keep_mask = make_keep_mask(voxels, edit_x, edit_y, edit_z)

    edited_voxels = dvd.edit_voxels(
        args.prompt,
        voxels,
        keep_mask=keep_mask,
        seed=args.seed,
        steps=args.dvd_steps,
    )

    edited_coords_path = os.path.join(args.output_dir, f"voxel64_{name}_edited_dis.npy")
    np.save(edited_coords_path, edited_voxels.coords_without_batch.numpy())
    export_cubified_voxels(edited_voxels, os.path.join(args.output_dir, f"{name}_edited_voxels.glb"))
    print(f"Saved edited DVD coords: {edited_coords_path}")

    if args.skip_stage2:
        return

    trellis = TrellisTextTo3DPipeline.from_pretrained(args.trellis_model)
    trellis.to(args.device)
    outputs = run_text_stage2_from_dvd_voxels(
        trellis,
        args.prompt,
        edited_voxels,
        seed=args.seed,
        slat_sampler_params={
            "steps": args.stage2_steps,
            "cfg_strength": args.stage2_cfg,
        },
    )

    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0])
    glb_path = os.path.join(args.output_dir, f"{name}_edited.glb")
    glb.export(glb_path)
    print(f"Saved edited GLB: {glb_path}")


if __name__ == "__main__":
    main()
