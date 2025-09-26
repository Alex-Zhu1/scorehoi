import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import trimesh
from PIL import Image
from hy3dgen.shapegen.utils import logger
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.rembg import BackgroundRemover

MAX_SEED = int(1e7)

def gen_save_folder(save_dir: str, max_size: int = 200) -> str:
    """Create a unique folder for saving outputs, removing oldest folder if limit is reached."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        dirs = [f for f in Path(save_dir).iterdir() if f.is_dir()]
        if len(dirs) >= max_size:
            oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
            shutil.rmtree(oldest_dir, ignore_errors=True)
            logger.info(f"Removed oldest folder: {oldest_dir}")
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{random.randint(0, 999999):06d}"
        new_folder = os.path.join(save_dir, timestamp)
        os.makedirs(new_folder, exist_ok=True)
        logger.info(f"Created new folder: {new_folder}")
        return new_folder
    except OSError as e:
        logger.error(f"Failed to create or manage save folder: {e}")
        raise

def export_mesh(mesh: trimesh.Trimesh, save_folder: str, textured: bool = False, file_type: str = 'glb') -> str:
    """Export mesh to specified format and folder."""
    if file_type not in ['glb', 'obj', 'ply', 'stl']:
        raise ValueError(f"Unsupported file type: {file_type}. Supported types are glb, obj, ply, stl.")
    
    file_name = f"{'textured' if textured else 'white'}_mesh.{file_type}"
    path = os.path.join(save_folder, file_name)
    try:
        if file_type in ['glb', 'obj']:
            mesh.export(path, include_normals=textured)
        else:
            mesh.export(path)
        logger.info(f"Exported mesh to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to export mesh to {path}: {e}")
        raise

def randomize_seed(seed: int, randomize: bool) -> int:
    """Return a randomized seed if requested."""
    return random.randint(0, MAX_SEED) if randomize else seed

def initialize_workers(
    model_path: str,
    subfolder: str,
    texgen_model_path: str,
    device: str,
    enable_texture: bool,
    enable_flashvdm: bool,
    low_vram_mode: bool,
    mc_algo: str,
    compile: bool
) -> Tuple[BackgroundRemover, Hunyuan3DDiTFlowMatchingPipeline, Optional[object]]:
    """Initialize shape generation and texture generation workers."""
    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        use_safetensors=True,
        device=device,
    )
    if enable_flashvdm:
        mc_algo = 'mc' if device in ['cpu', 'mps'] else mc_algo
        i23d_worker.enable_flashvdm(mc_algo=mc_algo)
    if compile:
        i23d_worker.compile()
    
    texgen_worker = None
    if enable_texture:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(texgen_model_path)
            if low_vram_mode:
                texgen_worker.enable_model_cpu_offload()
        except Exception as e:
            logger.warning(f"Failed to load texture generator: {e}")
            enable_texture = False
    
    return rmbg_worker, i23d_worker, texgen_worker

def initialize_t2i_worker(enable_t23d: bool, caption: Optional[str], image: Optional[Image.Image], device: str) -> Optional[object]:
    """Initialize text-to-image worker if needed."""
    if enable_t23d and caption and image is None:
        try:
            from hy3dgen.text2image import HunyuanDiTPipeline
            return HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=device)
        except Exception as e:
            logger.error(f"Failed to load text-to-image pipeline: {e}")
            raise ValueError("Text-to-image pipeline failed. Please check dependencies or provide an input image.")
    return None

def infer_shape(
    caption: Optional[str] = None,
    image: Optional[Image.Image] = None,
    mv_image_front: Optional[Image.Image] = None,
    mv_image_back: Optional[Image.Image] = None,
    mv_image_left: Optional[Image.Image] = None,
    mv_image_right: Optional[Image.Image] = None,
    steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 1234,
    octree_resolution: int = 256,
    check_box_rembg: bool = True,
    num_chunks: int = 200000,
    randomize: bool = False,
    model_path: str = 'tencent/Hunyuan3D-2mini',
    subfolder: str = 'hunyuan3d-dit-v2-mini-turbo',
    texgen_model_path: str = 'tencent/Hunyuan3D-2',
    device: str = 'cuda',
    enable_t23d: bool = False,
    enable_texture: bool = False,
    enable_flashvdm: bool = False,
    low_vram_mode: bool = False,
    mc_algo: str = 'mc',
    compile: bool = False,
    output_format: str = 'glb',
    save_dir: str = 'inference_output'
) -> Tuple[str, Optional[str], Dict, int]:
    """Generate 3D mesh from image or text input with optional texture."""
    # Validate inputs
    mv_mode = 'mv' in model_path.lower()
    if not mv_mode and image is None and caption is None:
        raise ValueError("Please provide either a caption or an image.")
    if mv_mode and all(img is None for img in [mv_image_front, mv_image_back, mv_image_left, mv_image_right]):
        raise ValueError("Please provide at least one view image.")

    # Prepare multi-view image dictionary
    if mv_mode:
        image = {
            view: img for view, img in [
                ('front', mv_image_front),
                ('back', mv_image_back),
                ('left', mv_image_left),
                ('right', mv_image_right)
            ] if img is not None
        }

    # Initialize workers
    rmbg_worker, i23d_worker, texgen_worker = initialize_workers(
        model_path, subfolder, texgen_model_path, device, enable_texture,
        enable_flashvdm, low_vram_mode, mc_algo, compile
    )
    t2i_worker = initialize_t2i_worker(enable_t23d, caption, image, device)

    # Prepare stats
    seed = randomize_seed(seed, randomize)
    stats = {
        'model': {
            'shapegen': f'{model_path}/{subfolder}',
            'texgen': f'{texgen_model_path}' if enable_texture and texgen_worker else 'Unavailable',
        },
        'params': {
            'caption': caption,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': int(octree_resolution),
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
            'output_format': output_format,
        },
        'time': {}
    }

    # Create save folder early for debugging images
    save_folder = gen_save_folder(save_dir)

    # Text-to-image generation if needed
    if image is None and t2i_worker:
        start_time = time.time()
        image = t2i_worker(caption)
        stats['time']['text2image'] = time.time() - start_time
        logger.info(f"Text-to-image generation took {stats['time']['text2image']:.2f} seconds")
        # Save debug image for text-to-image
        if check_box_rembg and image:
            image.save(os.path.join(save_folder, "text2image_rembg.png"))

    # Background removal
    if check_box_rembg:
        start_time = time.time()
        if mv_mode:
            for view, img in image.items():
                image[view] = rmbg_worker(img.convert('RGB'))
                image[view].save(os.path.join(save_folder, f"{view}_rembg.png"))  # Save for debugging
        else:
            image = rmbg_worker(image.convert('RGB'))
            image.save(os.path.join(save_folder, "rembg.png"))  # Save for debugging
        stats['time']['remove background'] = time.time() - start_time
        logger.info(f"Background removal took {stats['time']['remove background']:.2f} seconds")

    # Shape generation
    start_time = time.time()
    generator = torch.Generator().manual_seed(seed)
    outputs = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=int(octree_resolution),
        num_chunks=num_chunks,
        output_type='mesh'
    )
    stats['time']['shape generation'] = time.time() - start_time
    logger.info(f"Shape generation took {stats['time']['shape generation']:.2f} seconds")

    # Export mesh
    start_time = time.time()
    mesh = export_to_trimesh(outputs)[0]
    stats['time']['export to trimesh'] = time.time() - start_time
    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]
    logger.info(f"Mesh exported with {stats['number_of_faces']} faces and {stats['number_of_vertices']} vertices")

    # Save white mesh
    white_mesh_path = export_mesh(mesh, save_folder, textured=False, file_type=output_format)

    # Texture generation
    textured_mesh_path = None
    if enable_texture and texgen_worker:
        start_time = time.time()
        floater_remove_worker = FloaterRemover()
        degenerate_face_remove_worker = DegenerateFaceRemover()
        face_reduce_worker = FaceReducer()
        
        mesh = floater_remove_worker(mesh)
        mesh = degenerate_face_remove_worker(mesh)
        mesh = face_reduce_worker(mesh)
        
        textured_mesh = texgen_worker(mesh, image)
        stats['time']['texture generation'] = time.time() - start_time
        logger.info(f"Texture generation took {stats['time']['texture generation']:.2f} seconds")
        textured_mesh_path = export_mesh(textured_mesh, save_folder, textured=True, file_type=output_format)

    if low_vram_mode:
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache in low VRAM mode")

    return white_mesh_path, textured_mesh_path, stats, seed

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 3D mesh from image or text input.")
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2", help="Path to shape generation model")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-0", help="Subfolder for shape generation model")
    parser.add_argument("--texgen_model_path", type=str, default="tencent/Hunyuan3D-2", help="Path to texture generation model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (cuda, cpu, or mps)")
    parser.add_argument("--mc_algo", type=str, default="mc", help="Marching cubes algorithm")
    parser.add_argument("--cache-path", type=str, default="inference_output", help="Directory for output files")
    parser.add_argument("--enable_t23d", action="store_true", help="Enable text-to-3D generation")
    parser.add_argument("--enable_tex", action="store_true", help="Enable texture generation")
    parser.add_argument("--enable_flashvdm", action="store_true", help="Enable FlashVDM optimization")
    parser.add_argument("--compile", action="store_true", help="Compile the model for performance")
    parser.add_argument("--low_vram_mode", action="store_true", help="Enable low VRAM mode")
    parser.add_argument("--input_image", type=str, default=None, help="Path to input image")
    parser.add_argument("--mv_image_front", type=str, default=None, help="Path to front view image")
    parser.add_argument("--mv_image_back", type=str, default=None, help="Path to back view image")
    parser.add_argument("--mv_image_left", type=str, default=None, help="Path to left view image")
    parser.add_argument("--mv_image_right", type=str, default=None, help="Path to right view image")
    parser.add_argument("--caption", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.5, help="Guidance scale for generation")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--octree_resolution", type=int, default=256, help="Octree resolution for mesh generation")
    parser.add_argument("--num_chunks", type=int, default=8000, help="Number of chunks for processing")
    parser.add_argument("--check_box_rembg", action="store_true", default=True, help="Remove background (enabled by default)")
    parser.add_argument("--randomize", action="store_true", help="Randomize seed")
    parser.add_argument("--output-format", type=str, default="ply", choices=["glb", "obj", "ply", "stl"], help="Output mesh format")
    args = parser.parse_args()

    try:
        image = Image.open(args.input_image).convert("RGBA") if args.input_image else None
        mv_image_front = Image.open(args.mv_image_front).convert("RGBA") if args.mv_image_front else None
        mv_image_back = Image.open(args.mv_image_back).convert("RGBA") if args.mv_image_back else None
        mv_image_left = Image.open(args.mv_image_left).convert("RGBA") if args.mv_image_left else None
        mv_image_right = Image.open(args.mv_image_right).convert("RGBA") if args.mv_image_right else None

        white_mesh_path, textured_mesh_path, stats, seed = infer_shape(
            caption=args.caption,
            image=image,
            mv_image_front=mv_image_front,
            mv_image_back=mv_image_back,
            mv_image_left=mv_image_left,
            mv_image_right=mv_image_right,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            octree_resolution=args.octree_resolution,
            check_box_rembg=args.check_box_rembg,
            num_chunks=args.num_chunks,
            randomize=args.randomize,
            model_path=args.model_path,
            subfolder=args.subfolder,
            texgen_model_path=args.texgen_model_path,
            device=args.device,
            enable_t23d=args.enable_t23d,
            enable_texture=args.enable_tex,
            enable_flashvdm=args.enable_flashvdm,
            low_vram_mode=args.low_vram_mode,
            mc_algo=args.mc_algo,
            compile=args.compile,
            output_format=args.output_format,
            save_dir=args.cache_path
        )

        logger.info(f"White mesh saved at: {white_mesh_path}")
        if textured_mesh_path:
            logger.info(f"Textured mesh saved at: {textured_mesh_path}")
        logger.info(f"Stats: {stats}")
        logger.info(f"Seed used: {seed}")

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

"""
### 关键修改说明
1. **提前创建 `save_folder`**：
   - 将 `save_folder = gen_save_folder(save_dir)` 移到背景移除之前，确保在保存调试图像时文件夹已存在。
   - 原代码在形状生成后才创建 `save_folder`，这会导致你的调试代码无法访问 `save_folder`。

2. **整合调试图像保存逻辑**：
   - **多视图模式**：在 `mv_mode` 下，替换了原有的背景移除逻辑，添加了 `image[view].save(os.path.join(save_folder, f"{view}_rembg.png"))`，保存每个视图的背景移除结果（如 `front_rembg.png`）。
   - **单视图模式**：添加了保存单张背景移除图像的逻辑，保存为 `rembg.png`。
   - **文本到图像模式**：为 `enable_t23d` 模式添加了保存文本生成图像的逻辑（`text2image_rembg.png`），以便调试。

3. **保留原有功能**：
   - 背景移除默认启用（`check_box_rembg=True`）。
   - 所有其他功能（形状生成、纹理生成、网格导出等）保持不变。
   - 保留了类型提示、模块化初始化、错误处理等优化。

4. **调试图像输出**：
   - 多视图模式下，背景移除后的图像保存为 `<save_folder>/<view>_rembg.png`（如 `front_rembg.png`, `back_rembg.png`）。
   - 单视图模式下，保存为 `<save_folder>/rembg.png`。
   - 文本到图像模式下，保存为 `<save_folder>/text2image_rembg.png`（如果适用）。

### 使用示例

#### 1. **多视图模式**
运行以下命令，假设你有四张视图图像：
```bash
python infer_shape_with_debug_save.py \
    --mv_image_front /path/to/front.png \
    --mv_image_back /path/to/back.png \
    --mv_image_left /path/to/left.png \
    --mv_image_right /path/to/right.png \
    --model_path tencent/Hunyuan3D-2mini-mv \
    --output-format glb \
    --enable_tex
```
- **输出**：
  - 白色网格：`inference_output/<timestamp>/white_mesh.glb`
  - 纹理网格：`inference_output/<timestamp>/textured_mesh.glb`（如果 `enable_tex`）
  - 调试图像：`inference_output/<timestamp>/<view>_rembg.png`（如 `front_rembg.png`）
  - 日志：显示背景移除、形状生成等耗时。

#### 2. **单视图模式**
```bash
python infer_shape_with_debug_save.py \
    --input_image /path/to/image.png \
    --output-format glb \
    --enable_tex
```
- **输出**：
  - 白色网格：`inference_output/<timestamp>/white_mesh.glb`
  - 纹理网格：`inference_output/<timestamp>/textured_mesh.glb`（如果 `enable_tex`）
  - 调试图像：`inference_output/<timestamp>/rembg.png`

#### 3. **文本到 3D 模式**
```bash
python infer_shape_with_debug_save.py \
    --caption "a futuristic robot" \
    --enable_t23d \
    --output-format glb \
    --enable_tex
```
- **输出**：
  - 白色网格：`inference_output/<timestamp>/white_mesh.glb`
  - 纹理网格：`inference_output/<timestamp>/textured_mesh.glb`（如果 `enable_tex`）
  - 调试图像：`inference_output/<timestamp>/text2image_rembg.png`

#### 4. **检查调试图像**
- 打开 `inference_output/<timestamp>` 文件夹，查看保存的 `<view>_rembg.png` 或 `rembg.png` 文件，确认背景移除效果。
- 使用图像查看工具（如 Windows Photos、macOS Preview）或 Python 脚本加载图像：
  ```python
  from PIL import Image
  Image.open("inference_output/<timestamp>/front_rembg.png").show()
  ```

### 注意事项
1. **保存文件夹**：
   - 调试图像保存在与网格相同的 `save_folder` 中，文件名为 `<view>_rembg.png`（多视图）或 `rembg.png`（单视图）。
   - 确保 `save_dir`（默认 `inference_output`）有写权限。

2. **性能影响**：
   - 保存调试图像会增加磁盘 I/O，可能会略微降低性能。如果不需要调试，可以注释掉保存代码：
     ```python
     # image[view].save(os.path.join(save_folder, f"{view}_rembg.png"))  # Comment out to skip
     ```
   - 或者添加一个命令行参数控制是否保存调试图像（见下文）。

3. **错误处理**：
   - 如果保存调试图像失败（例如磁盘空间不足），`image.save` 可能抛出异常。当前代码未单独捕获此异常，建议添加：
     ```python
     try:
         image[view].save(os.path.join(save_folder, f"{view}_rembg.png"))
     except Exception as e:
         logger.warning(f"Failed to save debug image {view}_rembg.png: {e}")
     ```

4. **扩展建议**：
   - **添加调试开关**：在 `argparse` 中添加 `--save_debug_images` 参数，控制是否保存调试图像：
     ```python
     parser.add_argument("--save_debug_images", action="store_true", help="Save background-removed images for debugging")
     ```
     然后在 `infer_shape` 中检查：
     ```python
     if check_box_rembg and args.save_debug_images:
         # Save debug images
     ```
   - **调整图像格式**：默认保存为 PNG，可以添加参数支持其他格式（如 JPEG）以节省空间。

### 进一步调试
如果背景移除效果不佳：
- 检查 `BackgroundRemover` 的实现（`hy3dgen.rembg`），可能需要调整分割阈值或模型参数。
- 预处理输入图像（手动移除背景），并禁用 `check_box_rembg`：
  ```bash
  python infer_shape_with_debug_save.py --input_image /path/to/preprocessed_image.png --check_box_rembg False
  ```
- 查看调试图像（`<view>_rembg.png`），确认前景是否正确分割。

如果你遇到任何运行问题（如模型加载失败、背景移除效果不佳）或需要进一步优化（例如添加调试开关、调整图像保存格式），请提供具体细节，我可以为你提供更精确的解决方案！
"""