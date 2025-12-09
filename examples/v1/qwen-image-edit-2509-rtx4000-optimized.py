import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
import random
import string
import time

rank = 128  # you can also use rank=128 model to improve the quality

# print torch and cuda version
print(f"torch version: {torch.__version__}, cuda version: {torch.version.cuda}")

# print available GPUs
print(f"Available GPUs: {torch.cuda.device_count()}")

# print current device
print(f"Current device: {torch.cuda.current_device()}")

# print GPU name
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# print the precision
print(f"Using precision: {get_precision()}")

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509.safetensors"
)

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", transformer=transformer, torch_dtype=torch.bfloat16
)

# ============================================================================
# RTX 4000 Ada Generation Speed Optimizations (20GB VRAM)
# ============================================================================
# Strategy: Keep everything on GPU for maximum speed since we have 20GB VRAM
# No CPU offloading - that would slow us down!

# Move entire pipeline to GPU
pipeline.to("cuda")

# Enable VAE tiling only if needed for very large images
# This has minimal performance impact but helps with memory
pipeline.vae.enable_tiling()

# Enable SDPA (Scaled Dot Product Attention) - highly optimized on Ada Lovelace
# Ada architecture has excellent SDPA performance with bfloat16
try:
    # Use native PyTorch SDPA which is optimized for Ada Lovelace
    from diffusers.models.attention_processor import AttnProcessor2_0
    pipeline.transformer.set_attn_processor(AttnProcessor2_0())
    print("✓ Using PyTorch SDPA (optimized for Ada Lovelace architecture)")
except Exception as e:
    print(f"Note: {e}")

# Optional: torch.compile for additional speedup on Ada Lovelace (PyTorch 2.0+)
# Uncomment these lines for ~20-30% speedup after first (slower) compilation run
# Note: First inference will be slow due to compilation, subsequent runs will be faster
#
# if hasattr(torch, 'compile'):
#     print("Compiling model components with torch.compile...")
#     pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=False)
#     pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune")
#     print("✓ Model compiled (first run will be slower, subsequent runs faster)")

# Set optimal matmul precision for Ada Lovelace
# "high" gives good balance of speed and accuracy for bfloat16
torch.set_float32_matmul_precision('high')

# Enable TF32 for even faster matmul on Ada Lovelace (minimal accuracy impact)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("✓ TF32 enabled for faster computation on Ada Lovelace")

# Enable cuDNN benchmarking to find fastest convolution algorithms
torch.backends.cudnn.benchmark = True
print("✓ cuDNN auto-tuning enabled")

# Check memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
initial_vram = torch.cuda.memory_allocated() / 1024**3
print(f"\nInitial VRAM: {initial_vram:.2f} GB / 20 GB")

# Load test images
print("\nLoading images...")
image1 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png")
image1 = image1.convert("RGB")
image2 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png")
image2 = image2.convert("RGB")
image3 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png")
image3 = image3.convert("RGB")

prompt = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
inputs = {
    "image": [image1, image2, image3],
    "prompt": prompt,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "height": 720,
    "width": 1600,
}

# Warmup run (optional - helps stabilize timing measurements)
# Uncomment if you want more consistent timing results
# print("\nWarming up...")
# with torch.inference_mode():
#     _ = pipeline(**{**inputs, "num_inference_steps": 1})
# torch.cuda.synchronize()
# torch.cuda.reset_peak_memory_stats()

# Measure inference time
print("\nGenerating image...")
start_time = time.time()

with torch.inference_mode():  # Slightly faster than torch.no_grad()
    output = pipeline(**inputs)

torch.cuda.synchronize()  # Ensure all GPU operations complete
end_time = time.time()

output_image = output.images[0]
inference_time = end_time - start_time

# Performance metrics
print(f"\n{'='*60}")
print(f"Performance Metrics (RTX 4000 Ada Generation)")
print(f"{'='*60}")
print(f"Inference time: {inference_time:.2f} seconds")
print(f"Time per step: {inference_time/inputs['num_inference_steps']:.3f} seconds")

# Memory usage
peak_vram = torch.cuda.max_memory_allocated() / 1024**3
print(f"\nMemory Usage:")
print(f"Peak VRAM: {peak_vram:.2f} GB / 20 GB ({peak_vram/20*100:.1f}%)")
print(f"VRAM headroom: {20 - peak_vram:.2f} GB")

# Save output
random_letters = ''.join(random.choices(string.ascii_lowercase, k=4))
output_filename = f"/tmp/qwen-image-edit-2509-r{rank}-rtx4000-{random_letters}.png"
output_image.save(output_filename)
print(f"\nOutput saved to: {output_filename}")
print(f"{'='*60}")
