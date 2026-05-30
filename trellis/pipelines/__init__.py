from . import samplers
from .trellis_image_to_3d import TrellisImageTo3DPipeline
from .trellis_text_to_3d import TrellisTextTo3DPipeline
from .dvd_voxel import (
    DVDImageToVoxelPipeline,
    DVDTextToVoxelPipeline,
    DVDVoxelOutput,
    as_voxel_output,
    bsp_edit_cfg_schedule,
    coords_to_samples,
    dvd_coords_to_trellis_coords,
    export_cubified_voxels,
    image_cfg_schedule,
    load_dvd_denoiser,
    make_box_mask,
    normalize_cfg_schedule,
    run_image_stage2_from_dvd_voxels,
    run_text_stage2_from_dvd_voxels,
    samples_to_coords,
    text_cfg_schedule,
    trellis_coords_to_dvd_coords,
)


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
