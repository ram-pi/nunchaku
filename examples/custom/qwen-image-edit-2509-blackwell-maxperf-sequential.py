import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision
from datetime import datetime
import time

# Enable the fastest kernels on Blackwell (SM 120/121) and keep everything on GPU.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision("high")


def _enable_flash_sdp():
    """Try to force flash SDP; fall back quietly if unsupported."""
    try:
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
    except (AttributeError, RuntimeError) as err:
        print(f"Warning: flash SDP not available on this torch build/device: {err}")


_enable_flash_sdp()

# Optional knobs to push more throughput (may add a short warmup cost).
USE_COMPILE = True
NUM_IMAGES_PER_PROMPT = 1

device = torch.device("cuda")
rank = 128  # r128 offers better quality; Blackwell 96GB has room to spare
precision = get_precision(device=device)

model_path = (
    f"nunchaku-tech/nunchaku-qwen-image-edit-2509/"
    f"svdq-{precision}_r{rank}-qwen-image-edit-2509.safetensors"
)

# Load the 4-bit transformer and keep it resident on the GPU.
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
transformer.set_offload(False)

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=transformer,
    torch_dtype=torch.bfloat16,  # BF16 activations pair well with 4-bit weights on Blackwell
)

if USE_COMPILE:
    try:
        transformer = torch.compile(transformer, mode="max-autotune")
        pipeline.transformer = transformer
    except Exception as err:
        print(f"Warning: torch.compile unavailable, continuing without it: {err}")

pipeline.to(device)
pipeline.transformer.to(device)

# Load conditioning images once and reuse
image1 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png").convert("RGB")
image2 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png").convert("RGB")
image3 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png").convert("RGB")

# List of prompts to cycle through
prompts = [
    "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep.",
    "Make the man sit upright on the sofa while the puppy plays on the floor.",
    "Place the man on the sofa holding the puppy on his lap.",
    "Have the man lie on the sofa facing the puppy, which sleeps on the rug.",
]

# Run for 10 minutes (600 seconds)
duration_seconds = 600
start_time = time.time()
execution_count = 0
all_images = []

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

while time.time() - start_time < duration_seconds:
    # Cycle through prompts
    prompt = prompts[execution_count % len(prompts)]

    inputs = {
        "image": [image1, image2, image3],
        "prompt": prompt,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "height": 720,
        "width": 1600,
        "num_images_per_prompt": NUM_IMAGES_PER_PROMPT,
    }

    output = pipeline(**inputs)
    all_images.extend(output.images)
    execution_count += 1

    elapsed = time.time() - start_time
    print(f"Execution {execution_count} completed in {elapsed:.2f}s (prompt {execution_count % len(prompts)})")

total_time = time.time() - start_time
queries_per_minute = (execution_count / total_time) * 60

# Save all outputs
for idx, image in enumerate(all_images):
    image.save(f"qwen-image-edit-2509-blackwell-maxperf_{timestamp}_{idx}.png")

print(f"\n{'='*60}")
print(f"Blackwell Max-Perf Benchmark completed!")
print(f"Total executions: {execution_count}")
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Average time per execution: {total_time/execution_count:.2f}s")
print(f"Throughput: {execution_count/total_time:.4f} executions/sec")
print(f"Queries per minute (QPM): {queries_per_minute:.2f}")
print(f"Generated {len(all_images)} images (NUM_IMAGES_PER_PROMPT={NUM_IMAGES_PER_PROMPT})")
print(f"torch.compile enabled: {USE_COMPILE}")
print(f"{'='*60}")
