
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
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

def export_mesh(mesh: trimesh.Trimesh, save_folder: str, textured: bool = False, file_type: str = 'glb', index: int = 0) -> str:
    """Export mesh to specified format and folder with an index for multiple images."""
    if file_type not in ['glb', 'obj', 'ply', 'stl']:
        raise ValueError(f"Unsupported file type: {file_type}. Supported types are glb, obj, ply, stl.")
    
    file_name = f"{'textured' if textured else 'white'}_mesh_{index}.{file_type}"
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

def initialize_t2i_worker(enable_t23d: bool, caption: Optional[str], images: Optional[List[Image.Image]], device: str) -> Optional[object]:
    """Initialize text-to-image worker if needed."""
    if enable_t23d and caption and (images is None or len(images) == 0):
        try:
            from hy3dgen.text2image import HunyuanDiTPipeline
            return HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=device)
        except Exception as e:
            logger.error(f"Failed to load text-to-image pipeline: {e}")
            raise ValueError("Text-to-image pipeline failed. Please check dependencies or provide at least one input image.")
    return None

def infer_shape(
    caption: Optional[str] = None,
    images: Optional[List[Image.Image]] = None,
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
) -> List[Tuple[str, Optional[str], Dict, int]]:
    """Generate 3D meshes from a list of images or text input with optional texture."""
    # Validate inputs
    mv_mode = 'mv' in model_path.lower()
    if not mv_mode and images is None and caption is None:
        raise ValueError("Please provide either a caption or at least one image.")
    if mv_mode and all(img is None for img in [mv_image_front, mv_image_back, mv_image_left, mv_image_right]):
        raise ValueError("Please provide at least one view image.")
    if images is not None and len(images) == 0:
        raise ValueError("Image list cannot be empty.")

    # Prepare multi-view image dictionary
    if mv_mode:
        images = [{
            view: img for view, img in [
                ('front', mv_image_front),
                ('back', mv_image_back),
                ('left', mv_image_left),
                ('right', mv_image_right)
            ] if img is not None
        }]

    # Initialize workers
    rmbg_worker, i23d_worker, texgen_worker = initialize_workers(
        model_path, subfolder, texgen_model_path, device, enable_texture,
        enable_flashvdm, low_vram_mode, mc_algo, compile
    )
    t2i_worker = initialize_t2i_worker(enable_t23d, caption, images, device)

    # Create save folder early for debugging images
    save_folder = gen_save_folder(save_dir)

    # Prepare results
    results = []

    # Text-to-image generation if needed
    if images is None and t2i_worker:
        start_time = time.time()
        image = t2i_worker(caption)
        images = [image]  # Treat as single image
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
                'image_index': 0
            },
            'time': {}
        }
        stats['time']['text2image'] = time.time() - start_time
        logger.info(f"Text-to-image generation took {stats['time']['text2image']:.2f} seconds")
        if check_box_rembg:
            try:
                image.save(os.path.join(save_folder, "text2image_rembg.png"))
            except Exception as e:
                logger.warning(f"Failed to save debug image text2image_rembg.png: {e}")

    # Background removal
    start_time = time.time()
    processed_images = []
    if mv_mode:
        processed_images = images  # Multi-view images are dictionaries
        if check_box_rembg:
            for idx, image_dict in enumerate(images):
                processed_dict = {}
                for view, img in image_dict.items():
                    processed_dict[view] = rmbg_worker(img)  # Use BackgroundRemover (returns RGBA)
                    try:
                        processed_dict[view].save(os.path.join(save_folder, f"{view}_rembg_{idx}.png"))
                    except Exception as e:
                        logger.warning(f"Failed to save debug image {view}_rembg_{idx}.png: {e}")
                processed_images[idx] = processed_dict
    else:
        for idx, image in enumerate(images or []):
            processed_image = image
            if check_box_rembg:
                processed_image = rmbg_worker(image)  # Use BackgroundRemover (returns RGBA)
                try:
                    processed_image.save(os.path.join(save_folder, f"rembg_{idx}.png"))
                except Exception as e:
                    logger.warning(f"Failed to save debug image rembg_{idx}.png: {e}")
            processed_images.append(processed_image)
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
            'num_images': len(images or [])
        },
        'time': {}
    }
    stats['time']['remove background'] = time.time() - start_time
    logger.info(f"Background removal for {len(images or [])} images took {stats['time']['remove background']:.2f} seconds")

    # Shape generation
    start_time = time.time()
    generator = torch.Generator().manual_seed(seed)
    try:
        # Try passing the list of images directly
        outputs = i23d_worker(
            image=processed_images if not mv_mode else images[0],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=int(octree_resolution),
            num_chunks=num_chunks,
            output_type='mesh'
        )
        meshes = export_to_trimesh(outputs)
        if not isinstance(meshes, list):
            meshes = [meshes]  # Handle single mesh case
    except Exception as e:
        logger.warning(f"List processing failed: {e}. Falling back to single image processing.")
        # Fallback to processing each image individually
        meshes = []
        for idx, processed_image in enumerate(processed_images):
            output = i23d_worker(
                image=processed_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed + idx),
                octree_resolution=int(octree_resolution),
                num_chunks=num_chunks,
                output_type='mesh'
            )
            mesh = export_to_trimesh(output)[0]
            meshes.append(mesh)
    stats['time']['shape generation'] = time.time() - start_time
    logger.info(f"Shape generation for {len(meshes)} images took {stats['time']['shape generation']:.2f} seconds")

    # Export meshes and generate textures
    for idx, mesh in enumerate(meshes):
        stats['number_of_faces'] = mesh.faces.shape[0]
        stats['number_of_vertices'] = mesh.vertices.shape[0]
        logger.info(f"Mesh {idx} exported with {stats['number_of_faces']} faces and {stats['number_of_vertices']} vertices")

        # Save white mesh
        start_time = time.time()
        white_mesh_path = export_mesh(mesh, save_folder, textured=False, file_type=output_format, index=idx)
        stats['time']['export to trimesh'] = time.time() - start_time

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
            
            textured_mesh = texgen_worker(mesh, processed_images[idx] if not mv_mode else images[0])
            stats['time']['texture generation'] = time.time() - start_time
            logger.info(f"Texture generation for image {idx} took {stats['time']['texture generation']:.2f} seconds")
            textured_mesh_path = export_mesh(textured_mesh, save_folder, textured=True, file_type=output_format, index=idx)

        results.append((white_mesh_path, textured_mesh_path, stats.copy(), seed + idx))

    if low_vram_mode:
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache in low VRAM mode")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 3D mesh from multiple images or text input.")
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
    parser.add_argument("--input_image", type=str, default=None, help="Path to first input image")
    parser.add_argument("--input_image2", type=str, default=None, help="Path to second input image")
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
        images = []
        if args.input_image:
            images.append(Image.open(args.input_image).convert("RGBA"))
        if args.input_image2:
            images.append(Image.open(args.input_image2).convert("RGBA"))

        mv_image_front = Image.open(args.mv_image_front).convert("RGBA") if args.mv_image_front else None
        mv_image_back = Image.open(args.mv_image_back).convert("RGBA") if args.mv_image_back else None
        mv_image_left = Image.open(args.mv_image_left).convert("RGBA") if args.mv_image_left else None
        mv_image_right = Image.open(args.mv_image_right).convert("RGBA") if args.mv_image_right else None

        results = infer_shape(
            caption=args.caption,
            images=images,
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

        for idx, (white_mesh_path, textured_mesh_path, stats, seed) in enumerate(results):
            logger.info(f"Image {idx}: White mesh saved at: {white_mesh_path}")
            if textured_mesh_path:
                logger.info(f"Image {idx}: Textured mesh saved at: {textured_mesh_path}")
            logger.info(f"Image {idx}: Stats: {stats}")
            logger.info(f"Image {idx}: Seed used: {seed}")

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise
