import torch
import time
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from datetime import datetime

rank = 128  # you can also use rank=128 model to improve the quality

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509.safetensors"
)

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", transformer=transformer, torch_dtype=torch.bfloat16
)

pipeline.to("cuda")

image1 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png")
image1 = image1.convert("RGB")
image2 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png")
image2 = image2.convert("RGB")
image3 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png")
image3 = image3.convert("RGB")

# Example batch of prompts - edit this list to add your own prompts
prompts = [
    "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep.",
    "Make the man sit upright on the sofa while the puppy plays on the floor.",
    "Place the man on the sofa holding the puppy on his lap.",
    "Have the man lie on the sofa facing the puppy, which sleeps on the rug.",
    "Put the man on the sofa reading a book, puppy curled up beside him.",
    "Let the man nap on the sofa with the puppy dozing near his feet.",
    "Seat the man on the sofa with the puppy peeking from under a cushion.",
    "Man reclining on the sofa, puppy sleeping on a pillow on the floor.",
    "Man lounging on the sofa, puppy chewing a toy on the floor.",
    "Man resting on the sofa, puppy asleep by the coffee table.",
    "A man relaxing on a couch while a dog sleeps nearby.",
    "A person lying on a sofa with a puppy resting on the floor.",
    "A man sitting on a couch and a dog lying beside it.",
    "A man on a sofa with a puppy napping on the ground.",
    "A man reclining on a couch while a puppy sleeps on the floor.",
    "A man reading a book on a sofa with a dog lying next to him.",
    "A man taking a nap on a couch while a puppy rests on the floor.",
    "A man sitting comfortably on a sofa with his pet dog sleeping nearby.",
    "A man lounging on a couch as his dog naps on the floor.",
    "A man enjoying a quiet moment on a sofa with his puppy resting close by.",
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
    image.save(f"qwen-image-edit-2509-no-offloading_{timestamp}_{idx}.png")

print(f"\n{'='*60}")
print(f"Benchmark completed!")
print(f"Total executions: {execution_count}")
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Average time per execution: {total_time/execution_count:.2f}s")
print(f"Throughput: {execution_count/total_time:.4f} executions/sec")
print(f"Queries per minute (QPM): {queries_per_minute:.2f}")
print(f"Generated {len(all_images)} images")
print(f"{'='*60}")
