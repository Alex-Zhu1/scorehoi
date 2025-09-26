import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
from huggingface_hub import login
from lang_sam import LangSAM
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import os
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
import re
import json  # For loading custom prompts from a JSON file

# Download NLTK data (run once if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')

# Log in to Hugging Face
login(token="REMOVED")

# Initialize LangSAM model
model = LangSAM()

# Initialize BLIP for image captioning with safetensors
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Define directories
input_dir = "./input"  # Folder containing input images
output_base_dir = "./seg_output"  # Base output directory
hand_mask_dir = os.path.join(output_base_dir, "hand_mask")  # Subfolder for hand masks
object_mask_dir = os.path.join(output_base_dir, "object_mask")  # Subfolder for object masks
masked_image_dir = os.path.join(output_base_dir, "masked_image")  # Subfolder for masked images
results_dir = os.path.join(output_base_dir, "results")  # Subfolder for inpainting results

# Custom object prompts mapping: Multiple key-value pairs
custom_prompts = {
    "26_cropped_hoi_1.png": "Headphones",
    "325_cropped_hoi_1.png": "Scissors",
    "15_full_image_1.png": "Ice cream",
    "960_cropped_hoi_0.png": "House"
}  # Add more image-prompt pairs as needed

# Optional: Load custom prompts from a JSON file (uncomment to use)
# custom_prompts_file = "./custom_prompts.json"
# if os.path.exists(custom_prompts_file):
#     with open(custom_prompts_file, 'r') as f:
#         custom_prompts = json.load(f)
#     print(f"Loaded custom prompts from {custom_prompts_file}: {custom_prompts}")

# Create output directories if they don't exist
for directory in [output_base_dir, hand_mask_dir, object_mask_dir, masked_image_dir, results_dir]:
    os.makedirs(directory, exist_ok=True)

# Load reference mask (assuming it's the same for all images)
ref_mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")
ref_mask_mode = ref_mask.mode
ref_mask_width, ref_mask_height = ref_mask.size

# Load FluxKontextPipeline
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
).to("cuda")

# Function to generate object prompt using BLIP and NLTK
def generate_object_prompt(image_pil, image_file):
    # Check for custom prompt
    if image_file in custom_prompts:
        print(f"Using custom object prompt for {image_file}: {custom_prompts[image_file]}")
        return custom_prompts[image_file]
    
    # Fallback to BLIP if no custom prompt
    inputs = blip_processor(image_pil, return_tensors="pt").to("cuda")
    out = blip_model.generate(**inputs, max_length=50, num_return_sequences=1)
    caption = blip_processor.decode(out[0], skip_special_tokens=True).lower()
    print(f"Generated caption for image: {caption}")

    # Common words to filter (non-object words)
    common_words = {"a", "an", "the", "of", "in", "on", "with", "holding", "person", "man", "woman", "boy", "girl", "people", "is", "are", "and"}

    try:
        # Tokenize and perform POS tagging
        tokens = word_tokenize(caption)
        pos_tags = nltk.pos_tag(tokens)

        # Filter for nouns (NN: singular noun, NNS: plural noun) and exclude common words
        nouns = [word for word, pos in pos_tags if pos in ["NN", "NNS"] and word not in common_words and len(word) > 3]

        # Select the last noun as the primary object (or fallback to "object")
        object_prompt = nouns[-1] if nouns else "object"
    except Exception as e:
        print(f"NLTK tokenization failed: {str(e)}. Using fallback tokenization.")
        tokens = re.findall(r'\b\w+\b', caption.lower())
        nouns = [word for word in tokens if word not in common_words and len(word) > 3]
        object_prompt = nouns[-1] if nouns else "object"

    print(f"Selected object prompt: {object_prompt}")
    return object_prompt

# Get list of image files in input directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    try:
        # Load image
        image_path = os.path.join(input_dir, image_file)
        image_pil = Image.open(image_path).convert("RGB")
        image_width, image_height = image_pil.size
        image_name = Path(image_file).stem  # Get filename without extension

        # Detect hand in the image (fixed prompt)
        hand_text_prompt = "hand, arm"
        hand_results = model.predict([image_pil], [hand_text_prompt])

        # Detect object using dynamic prompt from BLIP or custom
        object_text_prompt = generate_object_prompt(image_pil, image_file)
        object_results = model.predict([image_pil], [object_text_prompt])

        # Extract and merge hand masks
        hand_masks = hand_results[0].get('masks', [])
        if hand_masks.size == 0:
            print(f"No hand masks detected for {image_file}. Skipping.")
            continue

        # Merge hand masks using logical OR
        merged_hand_mask = np.any(hand_masks, axis=0).astype(np.uint8)

        # Scale to match ref_mask's pixel value range
        if ref_mask_mode == "L":
            merged_hand_mask = merged_hand_mask * 255
        elif ref_mask_mode == "RGB":
            merged_hand_mask = np.stack([merged_hand_mask * 255] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported ref_mask mode: {ref_mask_mode}")

        # Convert to PIL image
        merged_hand_mask_pil = Image.fromarray(merged_hand_mask).convert(ref_mask_mode)

        # Save merged hand mask
        hand_mask_path = os.path.join(hand_mask_dir, f"{image_name}_hand.png")
        merged_hand_mask_pil.save(hand_mask_path, format="PNG")

        # Extract object mask
        object_masks = object_results[0].get('masks', [])
        if object_masks.size == 0:
            print(f"No object masks detected for {image_file} (prompt: {object_text_prompt}). Skipping.")
            continue

        # Take the first object mask
        object_mask = object_masks[0] if object_masks.shape[0] > 0 else None
        if object_mask is None:
            print(f"No valid object mask found for {image_file}. Skipping.")
            continue

        # Scale to match ref_mask's pixel value range
        if ref_mask_mode == "L":
            object_mask = (object_mask > 0).astype(np.uint8) * 255
        elif ref_mask_mode == "RGB":
            object_mask = np.stack([(object_mask > 0).astype(np.uint8) * 255] * 3, axis=-1)

        # Convert to PIL image
        object_mask_pil = Image.fromarray(object_mask).convert(ref_mask_mode)

        # Save object mask
        object_mask_path = os.path.join(object_mask_dir, f"{image_name}_object.png")
        object_mask_pil.save(object_mask_path, format="PNG")

        # Set hand region to white in the input image
        image_np = np.array(image_pil)
        mask_np = np.array(merged_hand_mask_pil.convert("L"))  # Convert mask to grayscale
        mask_np = (mask_np > 128).astype(np.uint8)  # Binarize mask

        # Apply mask to set hand region to white
        if ref_mask_mode == "RGB":
            mask_np = mask_np[:, :, 0] if mask_np.ndim == 3 else mask_np
        image_np[mask_np == 1] = [255, 255, 255]
        masked_image_pil = Image.fromarray(image_np).convert("RGB")

        # Save masked image
        masked_image_path = os.path.join(masked_image_dir, f"{image_name}_masked.png")
        masked_image_pil.save(masked_image_path, format="PNG")

        # Perform inpainting with merged hand mask
        result = pipe(
            image=image_pil,
            prompt="remove the hand, fingers, arm, human, people, body",
            guidance_scale=2.5
        ).images[0]

        # Ensure result matches image_pil's size and mode
        result = result.resize((image_width, image_height), Image.LANCZOS).convert("RGB")

        # Save the inpainting result
        result_path = os.path.join(results_dir, f"{image_name}_result.png")
        result.save(result_path, format="PNG")

        print(f"Processed {image_file} successfully (object prompt: {object_text_prompt}).")

    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")
        continue

print("**" * 10)
print("Batch processing completed.")