"""Model inference classes for VLM benchmarking."""

import torch
from abc import ABC, abstractmethod
from .utils import clear_gpu_memory, SPATIAL_SYSTEM_PROMPT


class BaseInference(ABC):
    """Base class for all VLM inference wrappers."""

    def __init__(self, model_path, max_new_tokens=128, **kwargs):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def generate(self, image, prompt):
        """Generate response for image + prompt."""

    @abstractmethod
    def cleanup(self):
        """Release model from GPU memory."""


class InternVLInference(BaseInference):
    """Inference wrapper for InternVL models."""

    def __init__(self, model_path, dtype="bfloat16", max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else torch.bfloat16
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            ).eval().cuda()
        except Exception:
            # Fallback without flash attention
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
            ).eval().cuda()

        self.generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    def generate(self, image, prompt):
        """Generate response for image + prompt."""
        full_prompt = f"{SPATIAL_SYSTEM_PROMPT}\n\n{prompt}"
        try:
            pixel_values = self._process_image(image)
            return self.model.chat(
                self.tokenizer, pixel_values, full_prompt, self.generation_config
            )
        except Exception:
            try:
                return self.model.chat(
                    self.tokenizer, image, full_prompt, self.generation_config
                )
            except Exception as e:
                print(f"  [InternVL Error]: {e}")
                return ""

    def _process_image(self, image):
        """Process image for InternVL."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu_memory()


class SmolVLMInference(BaseInference):
    """Inference wrapper for HuggingFaceTB/SmolVLM2-2.2B-Instruct."""

    def __init__(self, model_path, dtype="bfloat16", max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(model_path)
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else torch.bfloat16

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                _attn_implementation="flash_attention_2",
            ).to("cuda")
        except Exception:
            # Fallback without flash attention
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch_dtype
            ).to("cuda")

    def generate(self, image, prompt):
        """Generate response for image + prompt."""
        augmented_prompt = f"{SPATIAL_SYSTEM_PROMPT}\n\n{prompt}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": augmented_prompt},
                ]
            },
        ]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
                )

            input_len = inputs["input_ids"].shape[1]
            response = self.processor.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True
            )[0]
            return response.strip()
        except Exception as e:
            print(f"  [SmolVLM Error]: {e}")
            return ""

    def cleanup(self):
        del self.model
        del self.processor
        clear_gpu_memory()


class SAILInference(BaseInference):
    """Inference wrapper for BytedanceDouyinContent/SAIL-VL2-2B."""

    def __init__(self, model_path, dtype="bfloat16", max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        # from transformers import AutoTokenizer, AutoModel, AutoProcessor
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else torch.bfloat16
        device = torch.cuda.current_device()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch_dtype
        ).to(device)

    def generate(self, image, prompt):
        """Generate response for image + prompt."""
        augmented_prompt = f"{SPATIAL_SYSTEM_PROMPT}\n\n{prompt}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": augmented_prompt},
                ]
            },
        ]

        try:
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                images=image, text=text,
                return_tensors="pt", padding=True, truncation=True,
            ).to(self.model.device).to(torch.bfloat16)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )

            response = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            # Strip the input prompt from response if echoed
            if augmented_prompt in response:
                response = response.split(augmented_prompt)[-1]
            return response.strip()
        except Exception as e:
            print(f"  [SAIL Error]: {e}")
            return ""

    def cleanup(self):
        del self.model
        del self.tokenizer
        del self.processor
        clear_gpu_memory()


class LlamaVisionInference(BaseInference):
    """Inference wrapper for unsloth/Llama-3.2-11B-Vision-Instruct."""

    def __init__(self, model_path, load_in_4bit=True, max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        from unsloth import FastVisionModel

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(self.model)

    def generate(self, image, prompt):
        """Generate response for image + prompt.
        Note: Llama 3.2 Vision does not support system messages with images,
        so we prepend the system prompt into the user message.
        """
        augmented_prompt = f"{SPATIAL_SYSTEM_PROMPT}\n\n{prompt}"
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": augmented_prompt},
            ]},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                image, input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    temperature=1.5,
                    min_p=0.1,
                )

            input_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            print(f"  [Llama Error]: {e}")
            return ""

    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu_memory()


class SpatialBotInference(BaseInference):
    """Inference wrapper for RussRobin/SpatialBot-3B."""

    def __init__(self, model_path, dtype="float16", max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = 'cuda:0'
        self.np = np

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else torch.float16

        # Load model on CPU first
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Force the vision tower to load (it's lazy-loaded in SpatialBot)
        inner = self.model.get_model()
        if hasattr(inner, 'vision_tower'):
            vt = inner.vision_tower
            if hasattr(vt, 'load_model'):
                vt.load_model()
            inner.vision_tower = vt.to(device=self.device, dtype=torch_dtype)

        if hasattr(inner, 'mm_projector'):
            inner.mm_projector = inner.mm_projector.to(device=self.device, dtype=torch_dtype)

        # Move the entire model to device
        self.model = self.model.to(self.device)

        # Monkey-patch encode_images to guarantee device consistency
        _original_encode_images = self.model.encode_images
        device = self.device

        def _safe_encode_images(images):
            images = images.to(device=device, dtype=torch_dtype)
            features = _original_encode_images(images)
            return features.to(device=device)

        self.model.encode_images = _safe_encode_images

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def _create_dummy_depth(self, image):
        """Create a dummy depth map matching the input image size."""
        w, h = image.size
        from PIL import Image as PILImage
        gradient = self.np.linspace(0, 255, h, dtype=self.np.uint8)
        depth_array = self.np.tile(gradient[:, None], (1, w))
        three_channel = self.np.stack([depth_array] * 3, axis=-1)
        return PILImage.fromarray(three_channel, 'RGB')

    def generate(self, image, prompt):
        """Generate response for image + prompt using RGB + dummy depth."""
        augmented_prompt = f"{SPATIAL_SYSTEM_PROMPT}\n\n{prompt}"
        depth_image = self._create_dummy_depth(image)

        text = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: <image 1>\n<image 2>\n{augmented_prompt} ASSISTANT:"
        )
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        input_ids = torch.tensor(
            text_chunks[0] + [-201] + [-202] + text_chunks[1],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        try:
            image_tensor = self.model.process_images(
                [image, depth_image], self.model.config
            ).to(dtype=torch.float16, device=self.device).contiguous()

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=False,
                    repetition_penalty=1.0,
                )[0]

            response = self.tokenizer.decode(
                output_ids[input_ids.shape[1]:], skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            print(f"  [SpatialBot Error]: {e}")
            return ""

    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu_memory()


class QwenInference(BaseInference):
    """Inference wrapper for Qwen3-VL via Unsloth."""

    def __init__(self, model_path, load_in_4bit=True, max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        from unsloth import FastVisionModel

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(self.model)

    def generate(self, image, prompt):
        """Generate response for image + prompt."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SPATIAL_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                image, input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    temperature=1.5,
                    min_p=0.1,
                )

            input_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            print(f"  [Qwen Error]: {e}")
            return ""

    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu_memory()


# ============================================================
# FINETUNED (LoRA) MODEL INFERENCE CLASSES
# ============================================================


class InternVLLoRAInference(BaseInference):
    """Inference wrapper for InternVL with LoRA adapter."""

    def __init__(self, model_path, adapter_path, dtype="bfloat16", max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        from transformers import AutoTokenizer, AutoModel
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else torch.bfloat16

        try:
            base_model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            ).eval().cuda()
        except Exception:
            base_model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
            ).eval().cuda()

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()

        self.generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    def generate(self, image, prompt):
        """Generate response for image + prompt."""
        full_prompt = f"{SPATIAL_SYSTEM_PROMPT}\n\n{prompt}"
        try:
            pixel_values = self._process_image(image)
            return self.model.chat(
                self.tokenizer, pixel_values, full_prompt, self.generation_config
            )
        except Exception:
            try:
                return self.model.chat(
                    self.tokenizer, image, full_prompt, self.generation_config
                )
            except Exception as e:
                print(f"  [InternVL-LoRA Error]: {e}")
                return ""

    def _process_image(self, image):
        """Process image for InternVL."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu_memory()


class SmolVLMLoRAInference(BaseInference):
    """Inference wrapper for SmolVLM with LoRA adapter."""

    def __init__(self, model_path, adapter_path, dtype="bfloat16", max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel

        self.processor = AutoProcessor.from_pretrained(model_path)
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else torch.bfloat16

        try:
            base_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                _attn_implementation="flash_attention_2",
            ).to("cuda")
        except Exception:
            base_model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch_dtype
            ).to("cuda")

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()

    def generate(self, image, prompt):
        """Generate response for image + prompt."""
        augmented_prompt = f"{SPATIAL_SYSTEM_PROMPT}\n\n{prompt}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": augmented_prompt},
                ]
            },
        ]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
                )

            input_len = inputs["input_ids"].shape[1]
            response = self.processor.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True
            )[0]
            return response.strip()
        except Exception as e:
            print(f"  [SmolVLM-LoRA Error]: {e}")
            return ""

    def cleanup(self):
        del self.model
        del self.processor
        clear_gpu_memory()


class QwenLoRAInference(BaseInference):
    """Inference wrapper for Qwen3-VL with LoRA adapter via Unsloth."""

    def __init__(self, model_path, adapter_path, load_in_4bit=True, max_new_tokens=128, **kwargs):
        super().__init__(model_path, max_new_tokens)
        from unsloth import FastVisionModel

        # Load base model
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )

        # Load LoRA adapter on top of base model
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.merge_and_unload()
        FastVisionModel.for_inference(self.model)

    def generate(self, image, prompt):
        """Generate response for image + prompt."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SPATIAL_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                image, input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    temperature=1.5,
                    min_p=0.1,
                )

            input_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            print(f"  [Qwen-LoRA Error]: {e}")
            return ""

    def cleanup(self):
        del self.model
        del self.tokenizer
        clear_gpu_memory()


# Registry mapping model type strings to inference classes
MODEL_REGISTRY = {
    "internvl": InternVLInference,
    "smolvlm": SmolVLMInference,
    "sail": SAILInference,
    "llama": LlamaVisionInference,
    "spatialbot": SpatialBotInference,
    "qwen": QwenInference,
    "internvl_lora": InternVLLoRAInference,
    "smolvlm_lora": SmolVLMLoRAInference,
    "qwen_lora": QwenLoRAInference,
}


def get_inference_class(model_type):
    """Return the inference class for a given model type."""
    return MODEL_REGISTRY.get(model_type)
