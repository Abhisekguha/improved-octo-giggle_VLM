"""
InternVL2.5-1B Simple Inference Notebook
========================================
Standalone script to run InternVL inference on a single image.
"""

import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ============================================================
# Image Preprocessing
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_input, input_size=448, max_num=12):
    """Load image from path or PIL Image."""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError("image_input must be a file path or PIL Image")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ============================================================
# Load Model
# ============================================================

MODEL_PATH = "OpenGVLab/InternVL2_5-1B"

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
print("Model loaded!")


# ============================================================
# Inference Function
# ============================================================

def ask_image(image_input, question, max_new_tokens=256):
    """
    Ask a question about an image.

    Args:
        image_input: PIL Image or file path
        question: text question about the image
        max_new_tokens: max tokens to generate

    Returns:
        model response string
    """
    pixel_values = load_image(image_input, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    # Prepend <image> tag to question
    prompt = f"<image>\n{question}"

    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # --- Option 1: Load from file path ---
    # response = ask_image("path/to/your/image.jpg", "Describe this image.")

    # --- Option 2: Load from PIL Image ---
    # from PIL import Image
    # img = Image.open("path/to/image.jpg")
    # response = ask_image(img, "What objects are in this image?")

    # --- Option 3: Load from URL ---
    # import requests
    # from io import BytesIO
    # url = "https://example.com/image.jpg"
    # img = Image.open(BytesIO(requests.get(url).content))
    # response = ask_image(img, "Which object is closest to the camera?")

    # --- Option 4: From HuggingFace dataset ---
    # from datasets import load_dataset
    # ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    # sample = ds[0]
    # response = ask_image(sample['image'], sample['question'])

    print("\n--- InternVL2.5-1B Ready ---")
    print("Use: ask_image(image, question)")
    print("Example: response = ask_image('photo.jpg', 'How many chairs are in the scene?')")
