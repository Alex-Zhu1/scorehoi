import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup_mask.png")

repo_id = "black-forest-labs/FLUX.1-Fill-dev"
pipe = FluxFillPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16).to("cuda")

image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"output.png")