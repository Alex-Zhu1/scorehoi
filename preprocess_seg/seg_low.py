import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
from huggingface_hub import login
from lang_sam import LangSAM
import numpy as np

# Log in to Hugging Face
login(token="REMOVED")

# Initialize LangSAM model
model = LangSAM()

# Load image
image_pil = Image.open("./examples/325_cropped_hoi_1.png").convert("RGB")

# Load reference mask
ref_mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

# Get image and reference mask dimensions
image_width, image_height = image_pil.size
ref_mask_width, ref_mask_height = ref_mask.size
ref_mask_mode = ref_mask.mode

# Resize image and masks to a smaller resolution to reduce memory usage
target_size = (512, 512)  # Adjust based on your GPU memory (e.g., 512x512 for lower memory)
image_pil_resized = image_pil.resize(target_size, Image.LANCZOS)

# Detect hand in the resized image
hand_text_prompt = "hand, arm"
hand_results = model.predict([image_pil_resized], [hand_text_prompt])

# Detect scissors in the resized image
object_text_prompt = "Scissors"
object_results = model.predict([image_pil_resized], [object_text_prompt])

# Extract and merge hand masks
hand_masks = hand_results[0].get('masks', [])
if hand_masks.size == 0:
    raise ValueError("No hand masks detected. Please check LangSAM output.")

# Merge hand masks (shape: (N, H, W)) using logical OR
merged_hand_mask = np.any(hand_masks, axis=0).astype(np.uint8)

# Scale to match ref_mask's pixel value range (e.g., 0-255 for grayscale or RGB)
if ref_mask_mode == "L":
    merged_hand_mask = merged_hand_mask * 255  # Binary: 0 or 255
elif ref_mask_mode == "RGB":
    # If ref_mask is RGB, replicate grayscale mask across channels
    merged_hand_mask = np.stack([merged_hand_mask * 255] * 3, axis=-1)
else:
    raise ValueError(f"Unsupported ref_mask mode: {ref_mask_mode}")

# Convert to PIL image with ref_mask's mode
merged_hand_mask_pil = Image.fromarray(merged_hand_mask).convert(ref_mask_mode)

# Resize to match ref_mask dimensions if needed (but since we resized image, adjust accordingly)
if (merged_hand_mask_pil.width, merged_hand_mask_pil.height) != target_size:
    merged_hand_mask_pil = merged_hand_mask_pil.resize(target_size, Image.NEAREST)

# Save merged hand mask
merged_hand_mask_pil.save("merged_hand_mask.png", format="PNG")

# Extract object mask
object_masks = object_results[0].get('masks', [])
if object_masks.size == 0:
    raise ValueError("No object masks detected. Please check LangSAM output.")

# Take the first object mask
object_mask = object_masks[0] if object_masks.shape[0] > 0 else None
if object_mask is None:
    raise ValueError("No valid object mask found.")

# Scale to match ref_mask's pixel value range
if ref_mask_mode == "L":
    object_mask = (object_mask > 0).astype(np.uint8) * 255
elif ref_mask_mode == "RGB":
    object_mask = np.stack([(object_mask > 0).astype(np.uint8) * 255] * 3, axis=-1)

# Convert to PIL image with ref_mask's mode
object_mask_pil = Image.fromarray(object_mask).convert(ref_mask_mode)

# Resize to match target size
object_mask_pil = object_mask_pil.resize(target_size, Image.NEAREST)

# Save object mask
object_mask_pil.save("object_mask.png", format="PNG")

# Set hand region to white in the resized input image
image_np = np.array(image_pil_resized)
mask_np = np.array(merged_hand_mask_pil.convert("L"))  # Convert mask to grayscale for processing
mask_np = (mask_np > 128).astype(np.uint8)  # Binarize mask (1 for hand, 0 elsewhere)

# Apply mask to set hand region to white (255, 255, 255)
if image_np.ndim == 3 and image_np.shape[2] == 3:  # Ensure RGB
    image_np[mask_np == 1] = [255, 255, 255]  # Set hand region to white
masked_image_pil = Image.fromarray(image_np).convert("RGB")

# Visualize masked_image_pil by saving it
masked_image_pil.save("masked_image.png")

# Load FluxKontextPipeline with float16 for reduced memory
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.float16  # Use float16 for lower memory (assuming GPU supports it)
).to("cuda")

# Enable memory-efficient attention if xformers is installed
try:
    pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    print("xformers not installed; skipping memory-efficient attention.")

# Perform inpainting with merged hand mask on resized image
result = pipe(
    image=masked_image_pil,  # Use masked_image_pil
    prompt="remove the hand, fingers, arm, box",
    guidance_scale=2.5
).images[0]

# Ensure result matches original image size by resizing back
result = result.resize((image_width, image_height), Image.LANCZOS).convert("RGB")

# Save the inpainting result
result.save("flux-kontext-dev.png", format="PNG")

# Free memory
del pipe
torch.cuda.empty_cache()