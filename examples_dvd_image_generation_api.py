import argparse
import os

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
from PIL import Image

from trellis.pipelines import (
    DVDImageToVoxelPipeline,
    TrellisImageTo3DPipeline,
    export_cubified_voxels,
    run_image_stage2_from_dvd_voxels,
)
from trellis.utils import postprocessing_utils


def parse_args():
    parser = argparse.ArgumentParser(description="DVD image-to-voxel generation followed by TRELLIS stage 2.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--dvd-config", default="ckpts/dvd_img.json", help="DVD model config JSON.")
    parser.add_argument("--dvd-checkpoint", default="ckpts/dvd_img.safetensors", help="DVD safetensors checkpoint.")
    parser.add_argument("--output-dir", default="example_results", help="Directory for generated assets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image = Image.open(args.image)
    name = os.path.splitext(os.path.basename(args.image))[0]

    dvd = DVDImageToVoxelPipeline.from_files(
        args.dvd_config,
        args.dvd_checkpoint,
        device=args.device,
    )
    voxels = dvd.sample_voxels(image, seed=args.seed)
    voxel_coords_path = os.path.join(args.output_dir, f"voxel64_{name}_dis.npy")
    np.save(voxel_coords_path, voxels.coords_without_batch.numpy())
    export_cubified_voxels(voxels, os.path.join(args.output_dir, f"{name}_dvd_voxels.glb"))

    trellis = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    trellis.to(args.device)
    outputs = run_image_stage2_from_dvd_voxels(trellis, image, voxels, seed=args.seed)

    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0])
    glb.export(os.path.join(args.output_dir, f"{name}.glb"))
    print(f"Saved DVD coords: {voxel_coords_path}")


if __name__ == "__main__":
    main()
