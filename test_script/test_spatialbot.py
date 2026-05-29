"""
Standalone test script for RussRobin/SpatialBot-3B
Tests spatial understanding with a single RGB image (no depth map required).
"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np

# Disable warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# Config
device = 'cuda:0'
model_name = 'RussRobin/SpatialBot-3B'
offset_bos = 0

print(f"Device: {device}")
print(f"Loading model: {model_name}")

# Load model on CPU first, then move everything explicitly
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()

# Force the vision tower to load (it's lazy-loaded in SpatialBot)
inner = model.get_model()
if hasattr(inner, 'vision_tower'):
    vt = inner.vision_tower
    # Some models store it as a list or need explicit loading
    if hasattr(vt, 'load_model'):
        vt.load_model()
    inner.vision_tower = vt.to(device=device, dtype=torch.float16)

if hasattr(inner, 'mm_projector'):
    inner.mm_projector = inner.mm_projector.to(device=device, dtype=torch.float16)

# Move the entire model to device
model = model.to(device)

# Monkey-patch encode_images to guarantee device consistency
_original_encode_images = model.encode_images

def _safe_encode_images(images):
    images = images.to(device=device, dtype=torch.float16)
    features = _original_encode_images(images)
    return features.to(device=device)

model.encode_images = _safe_encode_images
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)
print("Model loaded successfully.")


def create_dummy_depth(image):
    """Create a dummy depth map (gray gradient) matching the input image size."""
    w, h = image.size
    gradient = np.linspace(0, 255, h, dtype=np.uint8)
    depth_array = np.tile(gradient[:, None], (1, w))
    three_channel = np.stack([depth_array] * 3, axis=-1)
    return Image.fromarray(three_channel, 'RGB')


def inference_with_depth(rgb_image, depth_image, prompt):
    """Run inference with both RGB and depth images."""
    text = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
    )
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
    input_ids = torch.tensor(
        text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    # Convert depth to 3-channel if grayscale
    channels = len(depth_image.getbands())
    if channels == 1:
        img = np.array(depth_image)
        height, width = img.shape
        three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
        three_channel_array[:, :, 0] = (img // 1024) * 4
        three_channel_array[:, :, 1] = (img // 32) * 8
        three_channel_array[:, :, 2] = (img % 32) * 8
        depth_image = Image.fromarray(three_channel_array, 'RGB')

    image_tensor = model.process_images(
        [rgb_image, depth_image], model.config
    ).to(dtype=torch.float16, device=device)

    # Ensure image_tensor is contiguous on cuda:0
    image_tensor = image_tensor.contiguous()

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=100,
        use_cache=False,
        repetition_penalty=1.0,
    )[0]

    response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response


def inference_single_image(rgb_image, prompt):
    """Run inference with just RGB (uses a dummy depth map)."""
    depth_image = create_dummy_depth(rgb_image)
    return inference_with_depth(rgb_image, depth_image, prompt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SpatialBot-3B with an image and question")
    parser.add_argument('--image', type=str, required=True, help='Path to the RGB image')
    parser.add_argument('--depth', type=str, default=None, help='Path to depth map (optional, generates dummy if not provided)')
    parser.add_argument('--question', type=str, required=True, help='Question to ask about the image')
    args = parser.parse_args()

    print("\nLoading image...")
    rgb_img = Image.open(args.image).convert('RGB')

    if args.depth:
        depth_img = Image.open(args.depth)
        response = inference_with_depth(rgb_img, depth_img, args.question)
    else:
        response = inference_single_image(rgb_img, args.question)

    print(f"\nQuestion: {args.question}")
    print(f"Response: {response}")
