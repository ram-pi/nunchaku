import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision
from datetime import datetime

# Enable the fastest kernels on Blackwell (SM 120/121) and keep everything on GPU.
torch.backends.cuda.matmul.allow_tf32 = True
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

pipeline.to(device)
pipeline.transformer.to(device)

image1 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png")
image1 = image1.convert("RGB")
image2 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png")
image2 = image2.convert("RGB")
image3 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png")
image3 = image3.convert("RGB")

prompt = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
inputs = {
    "image": [image1.convert("RGB"), image2.convert("RGB"), image3.convert("RGB")],
    "prompt": prompt,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "height": 720,
    "width": 1600,
}

output = pipeline(**inputs)

output_image = output.images[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_image.save(f"qwen-image-edit-2509-blackwell-r{rank}_{timestamp}.png")
