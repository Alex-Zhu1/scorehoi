import torch
from diffusers import FluxFillPipeline, FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
from huggingface_hub import login
from lang_sam import LangSAM
import numpy as np

# Log in to Hugging Face
# login(token="REMOVED")

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

# Convert ref_mask to numpy array for analysis
ref_mask_np = np.array(ref_mask)

# Detect hand in the image
hand_text_prompt = "hand, arm"
hand_results = model.predict([image_pil], [hand_text_prompt])

# Detect scissors in the image
object_text_prompt = "Scissors"
object_results = model.predict([image_pil], [object_text_prompt])

# Extract and merge hand masks
hand_masks = hand_results[0].get('masks', [])
if hand_masks.size == 0:
    raise ValueError("No hand masks detected. Please check LangSAM output.")

# Merge hand masks (shape: (2, 720, 720)) using logical OR
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

# Resize to match ref_mask dimensions if needed (commented out as per your code)
# if (merged_hand_mask_pil.width, merged_hand_mask_pil.height) != (ref_mask_width, ref_mask_height):
#     merged_hand_mask_pil = merged_hand_mask_pil.resize((ref_mask_width, ref_mask_height), Image.NEAREST)

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

# Resize to match ref_mask dimensions if needed (commented out as per your code)
# if (object_mask_pil.width, object_mask_pil.height) != (ref_mask_width, ref_mask_height):
#     object_mask_pil = object_mask_pil.resize((ref_mask_width, ref_mask_height), Image.NEAREST)

# Save object mask
object_mask_pil.save("object_mask.png", format="PNG")

# Set hand region to white in the input image
image_np = np.array(image_pil)
mask_np = np.array(merged_hand_mask_pil.convert("L"))  # Convert mask to grayscale for processing
mask_np = (mask_np > 128).astype(np.uint8)  # Binarize mask (1 for hand, 0 elsewhere)

# Apply mask to set hand region to white (255, 255, 255)
if ref_mask_mode == "RGB":
    # For RGB mask, assume all channels are identical and use one for masking
    mask_np = mask_np[:, :, 0] if mask_np.ndim == 3 else mask_np
image_np[mask_np == 1] = [255, 255, 255]  # Set hand region to white
masked_image_pil = Image.fromarray(image_np).convert("RGB")

# Visualize masked_image_pil by saving it
masked_image_pil.save("masked_image.png")

# # Load FluxFillPipeline (commented out as per your code)
# pipe = FluxFillPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-Fill-dev",
#     torch_dtype=torch.bfloat16
# ).to("cuda")
#
# # Perform inpainting with merged hand mask
# result = pipe(
#     prompt=f"an {object_text_prompt}, no hand, no fingers",
#     image=masked_image_pil,  # Fixed to use masked_image_pil
#     mask_image=merged_hand_mask_pil,
#     height=image_height,
#     width=image_width,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
#
# # Save the inpainting result
# result.save("flux-fill-dev.png")

# Load FluxKontextPipeline
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Perform inpainting with merged hand mask
result = pipe(
    image=image_pil,
    prompt="remove the hand, fingers, arm, box",
    guidance_scale=2.5
).images[0]

# Ensure result matches image_pil's size and mode
result = result.resize((image_width, image_height), Image.LANCZOS).convert("RGB")

# Save the inpainting result
result.save("flux-kontext-dev.png", format="PNG")
print("**"*10)

# result = pipe(
#     image=image_pil,submodules
#     prompt="remove the Scissors, box",
#     guidance_scale=2.5
# ).images[0]

# # Ensure result matches image_pil's size and mode
# result = result.resize((image_width, image_height), Image.LANCZOS).convert("RGB")

# # Save the inpainting result
# result.save("flux-kontext-dev-hand.png", format="PNG")