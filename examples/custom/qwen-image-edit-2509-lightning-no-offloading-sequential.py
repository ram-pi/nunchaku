import math
import time
from datetime import datetime

import pip

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# Scheduler config copied from Qwen Image Lightning reference
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

num_inference_steps = 8
rank = 32
model_path = (
    f"nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/"
    f"svdq-{get_precision()}_r{rank}-qwen-image-edit-2509-lightning-{num_inference_steps}steps-251115.safetensors"
)

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
)

# no offloading to keep everything on GPU for maximum performance
pipeline.to("cuda")
# if get_gpu_memory() > 18:
#     pipeline.enable_model_cpu_offload()
# else:
#     transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
#     pipeline._exclude_from_cpu_offload.append("transformer")
#     pipeline.enable_sequential_cpu_offload()

# Load conditioning images once
image1 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png").convert("RGB")
image2 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png").convert("RGB")
image3 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png").convert("RGB")

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
    prompt = prompts[execution_count % len(prompts)]

    inputs = {
        "image": [image1, image2, image3],
        "prompt": prompt,
        "true_cfg_scale": 1.0,
        "num_inference_steps": num_inference_steps,
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
    image.save(f"qwen-image-edit-2509-lightning-r{rank}-{num_inference_steps}steps_{timestamp}_{idx}.png")

print(f"\n{'='*60}")
print("Lightning Benchmark completed!")
print(f"Total executions: {execution_count}")
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Average time per execution: {total_time/execution_count:.2f}s")
print(f"Throughput: {execution_count/total_time:.4f} executions/sec")
print(f"Queries per minute (QPM): {queries_per_minute:.2f}")
print(f"Generated {len(all_images)} images")
print(f"{'='*60}")
