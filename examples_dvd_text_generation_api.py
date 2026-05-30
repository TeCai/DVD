import argparse
import os
import re

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from trellis.pipelines import (
    DVDTextToVoxelPipeline,
    TrellisTextTo3DPipeline,
    export_cubified_voxels,
    run_text_stage2_from_dvd_voxels,
)
from trellis.utils import postprocessing_utils


def slugify(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (slug[:max_length].strip("_") or "text_prompt")


def parse_args():
    parser = argparse.ArgumentParser(description="DVD text-to-voxel generation followed by TRELLIS stage 2.")
    parser.add_argument("--prompt", required=True, help="Input text prompt.")
    parser.add_argument("--name", default=None, help="Output asset name. Defaults to a slug of the prompt.")
    parser.add_argument("--dvd-config", default="ckpts/dvd_text.json", help="DVD text model config JSON.")
    parser.add_argument("--dvd-checkpoint", default="ckpts/dvd_text.safetensors", help="DVD text safetensors checkpoint.")
    parser.add_argument("--trellis-model", default="microsoft/TRELLIS-text-large", help="TRELLIS text pipeline.")
    parser.add_argument("--output-dir", default="example_results", help="Directory for generated assets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dvd-steps", type=int, default=256)
    parser.add_argument("--stage2-steps", type=int, default=25)
    parser.add_argument("--stage2-cfg", type=float, default=5.0)
    parser.add_argument("--skip-stage2", action="store_true", help="Only generate and save DVD voxels.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    name = args.name or slugify(args.prompt)

    dvd = DVDTextToVoxelPipeline.from_files(
        args.dvd_config,
        args.dvd_checkpoint,
        device=args.device,
    )
    voxels = dvd.sample_voxels(
        args.prompt,
        seed=args.seed,
        steps=args.dvd_steps,
    )

    voxel_coords_path = os.path.join(args.output_dir, f"voxel64_{name}_dis.npy")
    np.save(voxel_coords_path, voxels.coords_without_batch.numpy())
    export_cubified_voxels(voxels, os.path.join(args.output_dir, f"{name}_dvd_voxels.glb"))
    print(f"Saved DVD coords: {voxel_coords_path}")

    if args.skip_stage2:
        return

    trellis = TrellisTextTo3DPipeline.from_pretrained(args.trellis_model)
    trellis.to(args.device)
    outputs = run_text_stage2_from_dvd_voxels(
        trellis,
        args.prompt,
        voxels,
        seed=args.seed,
        slat_sampler_params={
            "steps": args.stage2_steps,
            "cfg_strength": args.stage2_cfg,
        },
    )

    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0])
    glb_path = os.path.join(args.output_dir, f"{name}.glb")
    glb.export(glb_path)
    print(f"Saved GLB: {glb_path}")


if __name__ == "__main__":
    main()
