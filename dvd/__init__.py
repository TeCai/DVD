"""Public DVD API built on top of TRELLIS internals."""

from trellis.pipelines import (
    DVDImageToVoxelPipeline,
    DVDTextToVoxelPipeline,
    DVDVoxelOutput,
    TrellisImageTo3DPipeline,
    TrellisTextTo3DPipeline,
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

__all__ = [
    "DVDImageToVoxelPipeline",
    "DVDTextToVoxelPipeline",
    "DVDVoxelOutput",
    "TrellisImageTo3DPipeline",
    "TrellisTextTo3DPipeline",
    "as_voxel_output",
    "bsp_edit_cfg_schedule",
    "coords_to_samples",
    "dvd_coords_to_trellis_coords",
    "export_cubified_voxels",
    "image_cfg_schedule",
    "load_dvd_denoiser",
    "make_box_mask",
    "normalize_cfg_schedule",
    "run_image_stage2_from_dvd_voxels",
    "run_text_stage2_from_dvd_voxels",
    "samples_to_coords",
    "text_cfg_schedule",
    "trellis_coords_to_dvd_coords",
    "load_image_pipeline",
    "load_image_edit_pipeline",
    "load_text_pipeline",
    "load_text_edit_pipeline",
]


def load_image_pipeline(repo_id_or_path: str, **kwargs) -> DVDImageToVoxelPipeline:
    """Load the DVD image generation pipeline from a local folder or Hugging Face repo."""
    return DVDImageToVoxelPipeline.from_pretrained(repo_id_or_path, variant="base", **kwargs)


def load_image_edit_pipeline(repo_id_or_path: str, **kwargs) -> DVDImageToVoxelPipeline:
    """Load the BSP fine-tuned DVD image editing pipeline."""
    return DVDImageToVoxelPipeline.from_pretrained(repo_id_or_path, variant="bsp", **kwargs)


def load_text_pipeline(repo_id_or_path: str, **kwargs) -> DVDTextToVoxelPipeline:
    """Load the DVD text generation pipeline from a local folder or Hugging Face repo."""
    return DVDTextToVoxelPipeline.from_pretrained(repo_id_or_path, variant="base", **kwargs)


def load_text_edit_pipeline(repo_id_or_path: str, **kwargs) -> DVDTextToVoxelPipeline:
    """Load the BSP fine-tuned DVD text editing pipeline."""
    return DVDTextToVoxelPipeline.from_pretrained(repo_id_or_path, variant="bsp", **kwargs)
