import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image
from datetime import datetime
import time

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# Configuration
rank = 32
num_inference_steps = 4
cfg_scale = 1.0
image_size = 1

# Select model path
# quen-image-edit-lightning
# model_path = f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit-lightningv1.0-{num_inference_steps}steps.safetensors"
# qwen-image-edit-2509-lightning
model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509-lightning-{num_inference_steps}steps-251115.safetensors"
# qwen-image-edit
# model_path = f"nunchaku-tech/nunchaku-qwen-image-edit/svdq-{get_precision()}_r{rank}-qwen-image-edit.safetensors"
# qwen-image-edit-2509
# model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509.safetensors"

# Base pipeline repo id
pipeline_repo_id = "Qwen/Qwen-Image-Edit"

# Log the chosen model configuration
print(
    f"Config -> transformer='{model_path}', pipeline='{pipeline_repo_id}', rank={rank}, steps={num_inference_steps}, precision={get_precision()}"
)

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)

pipeline = QwenImageEditPipeline.from_pretrained(
    pipeline_repo_id, transformer=transformer, torch_dtype=torch.bfloat16
)

if get_gpu_memory() > 18:
    pipeline.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )  # increase num_blocks_on_gpu if you have more VRAM
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()

image = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png")
image = image.convert("RGB")
prompt = "change the text to read '双截棍 Qwen Image Edit is here'"
inputs = {
    "image": image,
    "image_size": image_size,
    "prompt": prompt,
    "true_cfg_scale": cfg_scale,
    "negative_prompt": " ",
    "num_inference_steps": num_inference_steps,
}

# Run for 10 minutes (600 seconds), saving each result
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
duration_seconds = 600
start_time = time.time()
execution_count = 0

while time.time() - start_time < duration_seconds:
    iter_start = time.time()
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save(f"qwen-image-edit-r{rank}_{timestamp}_{execution_count}.png")
    execution_count += 1

    iter_elapsed = time.time() - iter_start
    total_elapsed = time.time() - start_time
    print(
        f"Execution {execution_count} | iter {iter_elapsed:.2f}s | total {total_elapsed:.2f}s"
    )

total_time = time.time() - start_time
if execution_count > 0:
    avg_time = total_time / execution_count
    qpm = (execution_count / total_time) * 60
else:
    avg_time = 0.0
    qpm = 0.0

print("=" * 60)
print("Benchmark completed!")
print(
    f"Model summary -> transformer='{model_path}', pipeline='{pipeline_repo_id}', rank={rank}, steps={num_inference_steps}"
)
print(f"Total executions: {execution_count}")
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Average time per execution: {avg_time:.2f}s")
print(f"Queries per minute (QPM): {qpm:.2f}")
print("=" * 60)
