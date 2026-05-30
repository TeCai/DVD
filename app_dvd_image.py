import os

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                return huggingface_hub.get_token()

            @staticmethod
            def save_token(token):
                return huggingface_hub.login(token=token, add_to_git_credential=False)

        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

import argparse
import shutil
from pathlib import Path

import gradio as gr
import gradio_client.utils as gradio_client_utils
import numpy as np
import torch
from gradio_litmodel3d import LitModel3D
from PIL import Image
from starlette.templating import Jinja2Templates

from dvd import (
    DVDImageToVoxelPipeline,
    TrellisImageTo3DPipeline,
    as_voxel_output,
    export_cubified_voxels,
    run_image_stage2_from_dvd_voxels,
)
from trellis.utils import postprocessing_utils


MAX_SEED = np.iinfo(np.int32).max
RESOLUTION = 64
ROOT_DIR = Path(__file__).resolve().parent
TMP_DIR = ROOT_DIR / "tmp" / "dvd_app"
TMP_DIR.mkdir(parents=True, exist_ok=True)

GEN_DVD_CONFIG = os.environ.get("DVD_GEN_CONFIG", "ckpts/dvd_img.json")
GEN_DVD_CKPT = os.environ.get("DVD_GEN_CKPT", "ckpts/dvd_img.safetensors")
EDIT_DVD_CONFIG = os.environ.get("DVD_EDIT_CONFIG", "ckpts/dvd_img_BSP_ft.json")
EDIT_DVD_CKPT = os.environ.get("DVD_EDIT_CKPT", "ckpts/dvd_img_BSP_ft.safetensors")
TRELLIS_IMAGE_MODEL = os.environ.get("TRELLIS_IMAGE_MODEL", "microsoft/TRELLIS-image-large")
DVD_MODEL_REPO = os.environ.get("DVD_MODEL_REPO")
DVD_MODEL_SUBFOLDER = os.environ.get("DVD_MODEL_SUBFOLDER") or None
DVD_MODEL_REVISION = os.environ.get("DVD_MODEL_REVISION") or None
DVD_MODEL_TOKEN = os.environ.get("DVD_MODEL_TOKEN") or os.environ.get("HF_TOKEN") or None
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_camera_position(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("DVD_VOXEL_CAMERA_POSITION must be three comma-separated numbers, e.g. -180,90,3")
    return tuple(parts)


VOXEL_CAMERA_POSITION = parse_camera_position(os.environ.get("DVD_VOXEL_CAMERA_POSITION", "-180,90,3"))


dvd_gen_pipeline = None
dvd_edit_pipeline = None
trellis_pipeline = None


_original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    if isinstance(schema, dict) and isinstance(schema.get("additionalProperties"), bool):
        schema = dict(schema)
        if schema["additionalProperties"]:
            schema["additionalProperties"] = {}
        else:
            schema.pop("additionalProperties")
    return _original_json_schema_to_python_type(schema, defs)


gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type


_original_template_response = Jinja2Templates.TemplateResponse


def _template_response_compat(self, *args, **kwargs):
    if args and isinstance(args[0], str):
        name = args[0]
        context = args[1] if len(args) > 1 else kwargs.pop("context", None)
        if isinstance(context, dict) and "request" in context:
            return _original_template_response(self, context["request"], name, context, *args[2:], **kwargs)
    return _original_template_response(self, *args, **kwargs)


Jinja2Templates.TemplateResponse = _template_response_compat


def list_asset_files(directory: str, suffixes: set[str]) -> list[Path]:
    path = ROOT_DIR / directory
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def asset_label(path: Path) -> str:
    return path.stem.replace("_", " ")


GENERATION_IMAGE_EXAMPLES = [
    (asset_label(path), str(path)) for path in list_asset_files("assets/example_image", IMAGE_EXTENSIONS)
]
EDIT_IMAGE_EXAMPLES = [
    (asset_label(path), str(path)) for path in list_asset_files("assets/example_image_edit", IMAGE_EXTENSIONS)
]
EDIT_VOXEL_EXAMPLES = [
    (asset_label(path), str(path)) for path in list_asset_files("assets/example_voxel_edit", {".npy", ".pt", ".pth"})
]


def voxel_viewer(label: str, exposure: float = 5.0, height: int = 300):
    return LitModel3D(
        label=label,
        exposure=exposure,
        height=height,
        camera_position=VOXEL_CAMERA_POSITION,
    )


def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def default_server_port():
    port = os.environ.get("GRADIO_SERVER_PORT")
    return int(port) if port else None


def load_dvd_pipelines(device: str):
    if DVD_MODEL_REPO:
        common_kwargs = {
            "device": device,
            "subfolder": DVD_MODEL_SUBFOLDER,
            "revision": DVD_MODEL_REVISION,
            "token": DVD_MODEL_TOKEN,
        }
        return (
            DVDImageToVoxelPipeline.from_pretrained(DVD_MODEL_REPO, variant="base", **common_kwargs),
            DVDImageToVoxelPipeline.from_pretrained(DVD_MODEL_REPO, variant="bsp", **common_kwargs),
        )
    return (
        DVDImageToVoxelPipeline.from_files(GEN_DVD_CONFIG, GEN_DVD_CKPT, device=device),
        DVDImageToVoxelPipeline.from_files(EDIT_DVD_CONFIG, EDIT_DVD_CKPT, device=device),
    )


def start_session(req: gr.Request):
    user_dir = TMP_DIR / str(req.session_hash)
    user_dir.mkdir(parents=True, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = TMP_DIR / str(req.session_hash)
    if user_dir.exists():
        shutil.rmtree(user_dir)


def session_path(req: gr.Request, name: str) -> str:
    user_dir = TMP_DIR / str(req.session_hash)
    user_dir.mkdir(parents=True, exist_ok=True)
    return str(user_dir / name)


def get_seed(randomize_seed: bool, seed: int) -> int:
    return int(np.random.randint(0, MAX_SEED)) if randomize_seed else int(seed)


def dvd_cfg_schedule(mode: str, constant: float, early: float, late: float, split: float):
    if mode == "Default schedule":
        return None
    if mode == "Constant":
        return float(constant)
    split = float(split)
    return lambda t: float(early) if t < split else float(late)


def dvd_sampler_kwargs(
    steps: int,
    cfg_mode: str,
    cfg_constant: float,
    cfg_early: float,
    cfg_late: float,
    cfg_split: float,
) -> dict:
    kwargs = {"steps": int(steps)}
    cfg_strength = dvd_cfg_schedule(cfg_mode, cfg_constant, cfg_early, cfg_late, cfg_split)
    if cfg_strength is not None:
        kwargs["cfg_strength"] = cfg_strength
    return kwargs


def slat_sampler_params(steps: int, cfg_strength: float) -> dict:
    return {
        "steps": int(steps),
        "cfg_strength": float(cfg_strength),
    }


def load_voxel_file(file, resolution: int = RESOLUTION):
    if file is None:
        raise gr.Error("Please upload a voxel coordinate file.")

    path = file.name if hasattr(file, "name") else str(file)
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            for key in ("coords", "voxels", "samples"):
                if key in data:
                    data = data[key]
                    break
    else:
        raise gr.Error(f"Unsupported voxel file type: {suffix}. Use .npy, .pt, or .pth.")
    return as_voxel_output(torch.as_tensor(data), resolution=resolution)


def load_image_from_path(path: str):
    if not path:
        raise gr.Error("Please select an example image.")
    return Image.open(path).convert("RGBA")


def load_generation_example(image_path: str):
    return load_image_from_path(image_path)


def load_edit_image_example(image_path: str):
    return load_image_from_path(image_path)


def load_edit_voxel_example(voxel_path: str, req: gr.Request):
    voxels = load_voxel_file(voxel_path)
    mesh_path = voxel_to_mesh(voxels, session_path(req, "example_edit_voxels.glb"))
    return voxels, mesh_path


def save_voxel_coords(voxels, path: str) -> str:
    output = as_voxel_output(voxels, resolution=RESOLUTION)
    np.save(path, output.coords_without_batch.numpy())
    return path


def voxel_to_mesh(voxels, path: str) -> str:
    return export_cubified_voxels(voxels, path, resolution=RESOLUTION)


def rotate_voxels(voxels, axis: str, req: gr.Request):
    if voxels is None:
        raise gr.Error("No editing voxels available. Upload voxels or transfer generated voxels first.")

    output = as_voxel_output(voxels, resolution=RESOLUTION)
    samples = output.samples.clone()
    axis_to_dims = {
        "x": (2, 3),
        "y": (1, 3),
        "z": (1, 2),
    }
    samples = torch.rot90(samples, k=1, dims=axis_to_dims[axis])
    rotated = as_voxel_output(samples, resolution=RESOLUTION)
    mesh_path = voxel_to_mesh(rotated, session_path(req, f"edit_voxels_rot_{axis}.glb"))
    return rotated, mesh_path


def rotate_x(voxels, req: gr.Request):
    return rotate_voxels(voxels, "x", req)


def rotate_y(voxels, req: gr.Request):
    return rotate_voxels(voxels, "y", req)


def rotate_z(voxels, req: gr.Request):
    return rotate_voxels(voxels, "z", req)


def build_edit_mask(
    use_1, x0_1, x1_1, y0_1, y1_1, z0_1, z1_1,
    use_2, x0_2, x1_2, y0_2, y1_2, z0_2, z1_2,
    use_3, x0_3, x1_3, y0_3, y1_3, z0_3, z1_3,
    batch_size: int = 1,
):
    boxes = [
        (use_1, x0_1, x1_1, y0_1, y1_1, z0_1, z1_1),
        (use_2, x0_2, x1_2, y0_2, y1_2, z0_2, z1_2),
        (use_3, x0_3, x1_3, y0_3, y1_3, z0_3, z1_3),
    ]
    edit_mask = torch.zeros((batch_size, RESOLUTION, RESOLUTION, RESOLUTION), dtype=torch.bool)
    any_box = False
    for use, x0, x1, y0, y1, z0, z1 in boxes:
        if not use:
            continue
        ranges = [int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)]
        x0, x1, y0, y1, z0, z1 = [max(0, min(RESOLUTION, v)) for v in ranges]
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
        edit_mask[:, x0:x1, y0:y1, z0:z1] = True
        any_box = True
    if not any_box:
        raise gr.Error("Enable at least one valid edit-mask box.")
    return edit_mask


def mask_inputs():
    return [
        box1_use, box1_x0, box1_x1, box1_y0, box1_y1, box1_z0, box1_z1,
        box2_use, box2_x0, box2_x1, box2_y0, box2_y1, box2_z0, box2_z1,
        box3_use, box3_x0, box3_x1, box3_y0, box3_y1, box3_z0, box3_z1,
    ]


def generate_voxels(
    image: Image.Image,
    seed: int,
    randomize_seed: bool,
    preprocess_image: bool,
    dvd_steps: int,
    dvd_cfg_mode: str,
    dvd_cfg_constant: float,
    dvd_cfg_early: float,
    dvd_cfg_late: float,
    dvd_cfg_split: float,
    req: gr.Request,
):
    if image is None:
        raise gr.Error("Please provide a generation image.")
    seed = get_seed(randomize_seed, seed)
    voxels = dvd_gen_pipeline.sample_voxels(
        image,
        seed=seed,
        preprocess_image=preprocess_image,
        **dvd_sampler_kwargs(dvd_steps, dvd_cfg_mode, dvd_cfg_constant, dvd_cfg_early, dvd_cfg_late, dvd_cfg_split),
    )
    mesh_path = voxel_to_mesh(voxels, session_path(req, "generated_voxels.glb"))
    npy_path = save_voxel_coords(voxels, session_path(req, "generated_voxel64_coords.npy"))
    torch.cuda.empty_cache()
    return voxels, mesh_path, npy_path, seed


def generation_stage2(
    image: Image.Image,
    voxels,
    seed: int,
    randomize_seed: bool,
    preprocess_image: bool,
    slat_steps: int,
    slat_cfg_strength: float,
    req: gr.Request,
):
    if image is None:
        raise gr.Error("Please provide the same generation image for TRELLIS stage 2.")
    if voxels is None:
        raise gr.Error("Generate voxels before running TRELLIS stage 2.")
    seed = get_seed(randomize_seed, seed)
    outputs = run_image_stage2_from_dvd_voxels(
        trellis_pipeline,
        image,
        voxels,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=preprocess_image,
        slat_sampler_params=slat_sampler_params(slat_steps, slat_cfg_strength),
    )
    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0])
    glb_path = session_path(req, "generated_stage2.glb")
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path, seed


def transfer_generation_to_editing(voxels, mesh_path):
    if voxels is None:
        raise gr.Error("No generated voxels to transfer.")
    return voxels, mesh_path


def load_edit_voxels(file, voxel_path: str, req: gr.Request):
    voxel_source = file if file is not None else voxel_path
    if voxel_source in (None, ""):
        raise gr.Error("Upload a voxel file or select a preloaded edit voxel.")
    voxels = load_voxel_file(voxel_source)
    mesh_path = voxel_to_mesh(voxels, session_path(req, "loaded_edit_voxels.glb"))
    return voxels, mesh_path


def visualize_edit_mask(
    voxels,
    use_1, x0_1, x1_1, y0_1, y1_1, z0_1, z1_1,
    use_2, x0_2, x1_2, y0_2, y1_2, z0_2, z1_2,
    use_3, x0_3, x1_3, y0_3, y1_3, z0_3, z1_3,
    req: gr.Request,
):
    if voxels is None:
        raise gr.Error("No editing voxels available.")
    output = as_voxel_output(voxels, resolution=RESOLUTION)
    edit_mask = build_edit_mask(
        use_1, x0_1, x1_1, y0_1, y1_1, z0_1, z1_1,
        use_2, x0_2, x1_2, y0_2, y1_2, z0_2, z1_2,
        use_3, x0_3, x1_3, y0_3, y1_3, z0_3, z1_3,
        batch_size=output.samples.shape[0],
    )
    perturbed = output.samples.clone()
    perturbed[edit_mask] = torch.randint(0, 2, perturbed[edit_mask].shape, dtype=perturbed.dtype)
    mask_preview = as_voxel_output(perturbed, resolution=RESOLUTION)
    mesh_path = voxel_to_mesh(mask_preview, session_path(req, "edit_mask_preview.glb"))
    return mesh_path


def run_editing(
    target_image: Image.Image,
    voxels,
    seed: int,
    randomize_seed: bool,
    preprocess_image: bool,
    dvd_steps: int,
    dvd_cfg_mode: str,
    dvd_cfg_constant: float,
    dvd_cfg_early: float,
    dvd_cfg_late: float,
    dvd_cfg_split: float,
    use_1, x0_1, x1_1, y0_1, y1_1, z0_1, z1_1,
    use_2, x0_2, x1_2, y0_2, y1_2, z0_2, z1_2,
    use_3, x0_3, x1_3, y0_3, y1_3, z0_3, z1_3,
    req: gr.Request,
):
    if target_image is None:
        raise gr.Error("Please provide a target image for editing.")
    if voxels is None:
        raise gr.Error("Upload voxels or transfer generated voxels first.")

    seed = get_seed(randomize_seed, seed)
    output = as_voxel_output(voxels, resolution=RESOLUTION)
    edit_mask = build_edit_mask(
        use_1, x0_1, x1_1, y0_1, y1_1, z0_1, z1_1,
        use_2, x0_2, x1_2, y0_2, y1_2, z0_2, z1_2,
        use_3, x0_3, x1_3, y0_3, y1_3, z0_3, z1_3,
        batch_size=output.samples.shape[0],
    )
    keep_mask = ~edit_mask
    edited = dvd_edit_pipeline.edit_voxels(
        target_image,
        output,
        keep_mask=keep_mask,
        seed=seed,
        preprocess_image=preprocess_image,
        **dvd_sampler_kwargs(dvd_steps, dvd_cfg_mode, dvd_cfg_constant, dvd_cfg_early, dvd_cfg_late, dvd_cfg_split),
    )
    mesh_path = voxel_to_mesh(edited, session_path(req, "edited_voxels.glb"))
    npy_path = save_voxel_coords(edited, session_path(req, "edited_voxel64_coords.npy"))
    torch.cuda.empty_cache()
    return edited, mesh_path, npy_path, seed


def editing_stage2(
    target_image: Image.Image,
    edited_voxels,
    seed: int,
    randomize_seed: bool,
    preprocess_image: bool,
    slat_steps: int,
    slat_cfg_strength: float,
    req: gr.Request,
):
    if target_image is None:
        raise gr.Error("Please provide the target image for TRELLIS stage 2.")
    if edited_voxels is None:
        raise gr.Error("Run editing before TRELLIS stage 2.")
    seed = get_seed(randomize_seed, seed)
    outputs = run_image_stage2_from_dvd_voxels(
        trellis_pipeline,
        target_image,
        edited_voxels,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=preprocess_image,
        slat_sampler_params=slat_sampler_params(slat_steps, slat_cfg_strength),
    )
    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0])
    glb_path = session_path(req, "edited_stage2.glb")
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path, seed


APP_CSS = """
#editing-three-col {
    align-items: flex-start;
}
#editing-three-col > div {
    min-width: 220px !important;
}
"""


with gr.Blocks(
    delete_cache=(600, 600),
    title="DVD + TRELLIS Voxel Generation and Editing",
    css=APP_CSS,
    fill_width=True,
) as demo:
    gr.Markdown(
        """
        ## DVD Voxel Generation and Editing
        DVD generates or edits a 64^3 voxel structure first. TRELLIS stage 2 is run only when you click the stage-2 button.
        """
    )

    generated_voxels_state = gr.State()
    edit_voxels_state = gr.State()
    edited_voxels_state = gr.State()

    with gr.Tab("Generation"):
        with gr.Row():
            with gr.Column():
                gen_example = gr.Dropdown(
                    choices=GENERATION_IMAGE_EXAMPLES,
                    label="Preloaded Generation Images",
                    value=None,
                    interactive=True,
                )
                gen_image = gr.Image(label="Condition Image", format="png", image_mode="RGBA", type="pil", height=300)
                with gr.Accordion("Generation Settings", open=False):
                    gen_seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                    gen_randomize = gr.Checkbox(value=True, label="Randomize seed")
                    gen_preprocess = gr.Checkbox(value=False, label="DVD preprocess image")
                    gen_dvd_steps = gr.Slider(1, 512, value=256, step=1, label="DVD voxel steps")
                    gen_dvd_cfg_mode = gr.Radio(
                        ["Default schedule", "Constant", "Two-stage"],
                        value="Default schedule",
                        label="DVD voxel CFG mode",
                    )
                    gen_dvd_cfg_constant = gr.Slider(
                        0.0,
                        5.0,
                        value=0.7,
                        step=0.05,
                        label="DVD voxel constant CFG",
                    )
                    with gr.Row():
                        gen_dvd_cfg_early = gr.Slider(0.0, 5.0, value=0.4, step=0.05, label="DVD CFG early t<0.5")
                        gen_dvd_cfg_late = gr.Slider(0.0, 5.0, value=0.7, step=0.05, label="DVD CFG late")
                    gen_dvd_cfg_split = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="DVD CFG switch time")
                    gen_stage2_preprocess = gr.Checkbox(value=True, label="TRELLIS preprocess image for stage 2")
                    gen_slat_steps = gr.Slider(1, 50, value=25, step=1, label="TRELLIS stage-2 steps")
                    gen_slat_cfg = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="TRELLIS stage-2 CFG")
                gen_btn = gr.Button("1. Generate DVD Voxels")
                gen_stage2_btn = gr.Button("2. Run TRELLIS Stage 2", interactive=True)
                transfer_btn = gr.Button("Move Generated Voxels To Editing")
            with gr.Column():
                gen_voxel_view = voxel_viewer("Generated / Cubified Voxels", exposure=5.0, height=320)
                gen_npy_download = gr.DownloadButton(label="Download Voxel Coords (.npy)", interactive=False)
                gen_stage2_view = LitModel3D(label="TRELLIS Stage 2 GLB", exposure=5.0, height=320)
                gen_glb_download = gr.DownloadButton(label="Download Stage 2 GLB", interactive=False)

    with gr.Tab("Editing"):
        with gr.Row(equal_height=False, elem_id="editing-three-col"):
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### 1. Source Voxels")
                edit_image_example = gr.Dropdown(
                    choices=EDIT_IMAGE_EXAMPLES,
                    label="Preloaded Edit Target Images",
                    value=None,
                    interactive=True,
                )
                edit_target_image = gr.Image(label="Target Image", format="png", image_mode="RGBA", type="pil", height=260)
                edit_voxel_example = gr.Dropdown(
                    choices=EDIT_VOXEL_EXAMPLES,
                    label="Preloaded Edit Voxels",
                    value=None,
                    interactive=True,
                )
                edit_file = gr.File(label="Upload Voxel Coords (.npy, .pt, .pth)")
                load_edit_btn = gr.Button("Load Selected / Uploaded Voxels")
                with gr.Row():
                    rot_x_btn = gr.Button("Rotate X 90")
                    rot_y_btn = gr.Button("Rotate Y 90")
                    rot_z_btn = gr.Button("Rotate Z 90")
                edit_voxel_view = voxel_viewer("Current Editing Voxels", exposure=10.0, height=300)

            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Edit Mask Boxes\nEach enabled box is an edit region. The union of enabled boxes is regenerated.")
                with gr.Accordion("Box 1", open=True):
                    box1_use = gr.Checkbox(value=True, label="Use box 1")
                    with gr.Row():
                        box1_x0 = gr.Slider(0, RESOLUTION, value=0, step=1, label="x0")
                        box1_x1 = gr.Slider(0, RESOLUTION, value=RESOLUTION, step=1, label="x1")
                    with gr.Row():
                        box1_y0 = gr.Slider(0, RESOLUTION, value=0, step=1, label="y0")
                        box1_y1 = gr.Slider(0, RESOLUTION, value=RESOLUTION, step=1, label="y1")
                    with gr.Row():
                        box1_z0 = gr.Slider(0, RESOLUTION, value=32, step=1, label="z0")
                        box1_z1 = gr.Slider(0, RESOLUTION, value=RESOLUTION, step=1, label="z1")
                with gr.Accordion("Box 2", open=False):
                    box2_use = gr.Checkbox(value=False, label="Use box 2")
                    with gr.Row():
                        box2_x0 = gr.Slider(0, RESOLUTION, value=0, step=1, label="x0")
                        box2_x1 = gr.Slider(0, RESOLUTION, value=RESOLUTION, step=1, label="x1")
                    with gr.Row():
                        box2_y0 = gr.Slider(0, RESOLUTION, value=0, step=1, label="y0")
                        box2_y1 = gr.Slider(0, RESOLUTION, value=RESOLUTION, step=1, label="y1")
                    with gr.Row():
                        box2_z0 = gr.Slider(0, RESOLUTION, value=0, step=1, label="z0")
                        box2_z1 = gr.Slider(0, RESOLUTION, value=16, step=1, label="z1")
                with gr.Accordion("Box 3", open=False):
                    box3_use = gr.Checkbox(value=False, label="Use box 3")
                    with gr.Row():
                        box3_x0 = gr.Slider(0, RESOLUTION, value=0, step=1, label="x0")
                        box3_x1 = gr.Slider(0, RESOLUTION, value=RESOLUTION, step=1, label="x1")
                    with gr.Row():
                        box3_y0 = gr.Slider(0, RESOLUTION, value=0, step=1, label="y0")
                        box3_y1 = gr.Slider(0, RESOLUTION, value=RESOLUTION, step=1, label="y1")
                    with gr.Row():
                        box3_z0 = gr.Slider(0, RESOLUTION, value=16, step=1, label="z0")
                        box3_z1 = gr.Slider(0, RESOLUTION, value=32, step=1, label="z1")
                preview_mask_btn = gr.Button("Preview Edit Region By Perturbing")
                edit_mask_view = voxel_viewer("Edit Region Preview", exposure=5.0, height=300)

            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### 3. Edited Result")
                with gr.Accordion("Editing Settings", open=False):
                    edit_seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                    edit_randomize = gr.Checkbox(value=True, label="Randomize seed")
                    edit_preprocess = gr.Checkbox(value=True, label="DVD preprocess target image")
                    edit_dvd_steps = gr.Slider(1, 512, value=128, step=1, label="DVD edit steps")
                    edit_dvd_cfg_mode = gr.Radio(
                        ["Default schedule", "Constant", "Two-stage"],
                        value="Default schedule",
                        label="DVD edit CFG mode",
                    )
                    edit_dvd_cfg_constant = gr.Slider(
                        0.0,
                        5.0,
                        value=0.45,
                        step=0.05,
                        label="DVD edit constant CFG",
                    )
                    with gr.Row():
                        edit_dvd_cfg_early = gr.Slider(0.0, 5.0, value=0.45, step=0.05, label="DVD CFG early t<0.5")
                        edit_dvd_cfg_late = gr.Slider(0.0, 5.0, value=0.45, step=0.05, label="DVD CFG late")
                    edit_dvd_cfg_split = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="DVD CFG switch time")
                    edit_stage2_preprocess = gr.Checkbox(value=True, label="TRELLIS preprocess target image for stage 2")
                    edit_slat_steps = gr.Slider(1, 50, value=25, step=1, label="TRELLIS stage-2 steps")
                    edit_slat_cfg = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="TRELLIS stage-2 CFG")
                edit_btn = gr.Button("Run DVD Editing")
                edit_stage2_btn = gr.Button("Run TRELLIS Stage 2")
                edited_voxel_view = voxel_viewer("Edited / Cubified Voxels", exposure=5.0, height=300)
                edited_npy_download = gr.DownloadButton(label="Download Edited Voxel Coords (.npy)", interactive=False)
                edit_stage2_view = LitModel3D(label="Edited TRELLIS Stage 2 GLB", exposure=5.0, height=300)
                edited_glb_download = gr.DownloadButton(label="Download Edited Stage 2 GLB", interactive=False)

    demo.load(start_session)
    demo.unload(end_session)

    gen_example.change(load_generation_example, inputs=[gen_example], outputs=[gen_image])
    edit_image_example.change(load_edit_image_example, inputs=[edit_image_example], outputs=[edit_target_image])
    edit_voxel_example.change(
        load_edit_voxel_example,
        inputs=[edit_voxel_example],
        outputs=[edit_voxels_state, edit_voxel_view],
    )

    gen_btn.click(
        generate_voxels,
        inputs=[
            gen_image,
            gen_seed,
            gen_randomize,
            gen_preprocess,
            gen_dvd_steps,
            gen_dvd_cfg_mode,
            gen_dvd_cfg_constant,
            gen_dvd_cfg_early,
            gen_dvd_cfg_late,
            gen_dvd_cfg_split,
        ],
        outputs=[generated_voxels_state, gen_voxel_view, gen_npy_download, gen_seed],
    ).then(lambda: gr.DownloadButton(interactive=True), outputs=[gen_npy_download])

    gen_stage2_btn.click(
        generation_stage2,
        inputs=[
            gen_image,
            generated_voxels_state,
            gen_seed,
            gen_randomize,
            gen_stage2_preprocess,
            gen_slat_steps,
            gen_slat_cfg,
        ],
        outputs=[gen_stage2_view, gen_glb_download, gen_seed],
    ).then(lambda: gr.DownloadButton(interactive=True), outputs=[gen_glb_download])

    transfer_btn.click(
        transfer_generation_to_editing,
        inputs=[generated_voxels_state, gen_voxel_view],
        outputs=[edit_voxels_state, edit_voxel_view],
    )

    load_edit_btn.click(
        load_edit_voxels,
        inputs=[edit_file, edit_voxel_example],
        outputs=[edit_voxels_state, edit_voxel_view],
    )
    rot_x_btn.click(rotate_x, inputs=[edit_voxels_state], outputs=[edit_voxels_state, edit_voxel_view])
    rot_y_btn.click(rotate_y, inputs=[edit_voxels_state], outputs=[edit_voxels_state, edit_voxel_view])
    rot_z_btn.click(rotate_z, inputs=[edit_voxels_state], outputs=[edit_voxels_state, edit_voxel_view])

    preview_mask_btn.click(
        visualize_edit_mask,
        inputs=[edit_voxels_state] + mask_inputs(),
        outputs=[edit_mask_view],
    )

    edit_btn.click(
        run_editing,
        inputs=[
            edit_target_image,
            edit_voxels_state,
            edit_seed,
            edit_randomize,
            edit_preprocess,
            edit_dvd_steps,
            edit_dvd_cfg_mode,
            edit_dvd_cfg_constant,
            edit_dvd_cfg_early,
            edit_dvd_cfg_late,
            edit_dvd_cfg_split,
        ] + mask_inputs(),
        outputs=[edited_voxels_state, edited_voxel_view, edited_npy_download, edit_seed],
    ).then(lambda: gr.DownloadButton(interactive=True), outputs=[edited_npy_download])

    edit_stage2_btn.click(
        editing_stage2,
        inputs=[
            edit_target_image,
            edited_voxels_state,
            edit_seed,
            edit_randomize,
            edit_stage2_preprocess,
            edit_slat_steps,
            edit_slat_cfg,
        ],
        outputs=[edit_stage2_view, edited_glb_download, edit_seed],
    ).then(lambda: gr.DownloadButton(interactive=True), outputs=[edited_glb_download])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--server-port", type=int, default=default_server_port())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device(args.device)

    dvd_gen_pipeline, dvd_edit_pipeline = load_dvd_pipelines(device)
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_IMAGE_MODEL)
    trellis_pipeline.to(device)

    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_api=False,
    )
