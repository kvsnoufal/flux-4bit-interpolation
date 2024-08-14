import os
import torch
import numpy as np
from PIL import Image
from model import T5EncoderModel, FluxTransformer2DModel
from diffusers import FluxPipeline
from IPython.display import display

# Constants for output directory, image dimensions, and interpolation steps
OUTDIR = 'outputs'
os.makedirs(OUTDIR, exist_ok=True)  # Create output directory if it doesn't exist
HEIGHT = 512
WIDTH = 512
NUM_INTERPOLATION_STEPS = 20
# List of prompts representing different stages of Dubai's evolution
prompts = [
    "Old Dubai Desert Village with tents and wells, aerial view",
    "Old Dubai City, aerial view",
    "Dubai City with towering skyscrapers, aerial view",
    "Super Futuristic Dubai City with glass buildings, vertical gardens, flying cars, and holographic displays, aerial view",
]

# Function to load models
def load_models():
    """
    Load the necessary models (text encoder, transformer, and pipeline) with specified configurations.

    Returns:
        pipe (FluxPipeline): The initialized pipeline loaded onto the GPU.
    """
    text_encoder = T5EncoderModel.from_pretrained(
        "HighCWu/FLUX.1-dev-4bit",
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        "HighCWu/FLUX.1-dev-4bit",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder_2=text_encoder,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    return pipe.to('cuda')  # Move pipeline to GPU

pipe = load_models()
pipe.remove_all_hooks()  # Remove any existing hooks from the pipeline

# Function to set up memory management hooks
def setup_memory_hooks():
    """
    Set up hooks to manage GPU memory by clearing cache and offloading to CPU when necessary.

    Returns:
        tuple: Hooks registered to manage memory during model execution.
    """
    def clean_hook(module, args, *rest_args):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    hook1 = pipe.transformer.register_forward_pre_hook(clean_hook)
    hook2 = pipe.transformer.register_forward_hook(clean_hook)
    
    def cpu_offload_hook(module, args):
        pipe.transformer.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    hook3 = pipe.vae.decoder.register_forward_pre_hook(cpu_offload_hook)
    
    return hook1, hook2, hook3

hook1, hook2, hook3 = setup_memory_hooks()



# Function to encode a text prompt into latent representations
def encode_prompt(pipe, prompt):
    """
    Encode a text prompt into embeddings that the model can process.

    Args:
        pipe (FluxPipeline): The initialized pipeline.
        prompt (str): The text prompt to be encoded.

    Returns:
        tuple: Prompt embeddings and pooled prompt embeddings.
    """
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe._execution_device,
            max_sequence_length=512
        )
    return prompt_embeds, pooled_prompt_embeds

# Function for spherical linear interpolation (slerp) between two vectors
def slerp(val, low, high):
    """
    Perform spherical linear interpolation (slerp) between two latent vectors.

    Args:
        val (float): The interpolation factor (0.0 to 1.0).
        low (torch.Tensor): The starting latent vector.
        high (torch.Tensor): The ending latent vector.

    Returns:
        torch.Tensor: The interpolated latent vector.
    """
    omega = torch.acos(torch.sum(low / torch.norm(low) * high / torch.norm(high)))
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # Fall back to linear interpolation if omega is zero
    return (torch.sin((1.0 - val) * omega) / so * low +
            torch.sin(val * omega) / so * high)

# Function to prepare latent space for image generation
def prepare_latents(pipe):
    """
    Prepare latent space for generating images.

    Args:
        pipe (FluxPipeline): The initialized pipeline.

    Returns:
        torch.Tensor: The prepared latent tensor.
    """
    num_channels_latents = pipe.transformer.config.in_channels // 4
    return pipe.prepare_latents(
        1,
        num_channels_latents,
        HEIGHT,
        WIDTH,
        torch.bfloat16,
        pipe._execution_device,
        torch.Generator("cpu").manual_seed(0),
        None,
    )[0]

latents = prepare_latents(pipe)

# Function to generate interpolated images between consecutive prompts
def generate_interpolated_images(prompts, latents):
    """
    Generate a series of interpolated images between consecutive prompts.

    Args:
        prompts (list): A list of text prompts.
        latents (torch.Tensor): The initial latent tensor.

    Returns:
        list: A list of generated PIL images.
    """
    images = []
    image_index = 0
    
    for i in range(len(prompts) - 1):
        # Encode the prompts into embeddings
        prompt_embeds_1, pooled_prompt_embeds_1 = encode_prompt(pipe, prompts[i])
        prompt_embeds_2, pooled_prompt_embeds_2 = encode_prompt(pipe, prompts[i + 1])
        
        for alpha in np.linspace(0, 1, NUM_INTERPOLATION_STEPS):
            # Interpolate between the embeddings
            interpolated_prompt_embeds = slerp(alpha, prompt_embeds_1, prompt_embeds_2)
            interpolated_pooled_prompt_embeds = slerp(alpha, pooled_prompt_embeds_1, pooled_prompt_embeds_2)
            
            # Generate the image using the interpolated embeddings
            image = pipe(
                prompt_embeds=interpolated_prompt_embeds,
                pooled_prompt_embeds=interpolated_pooled_prompt_embeds,
                height=HEIGHT,
                width=WIDTH,
                guidance_scale=0,
                output_type="pil",
                num_inference_steps=25,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0),
                latents=latents
            ).images[0]
            
            pipe.transformer.to('cuda')  # Ensure transformer is back on GPU
            image.save(os.path.join(OUTDIR, f"interpolated_image_{image_index}.png"))
            images.append(image)
            image_index += 1
    
    return images

interpolated_images = generate_interpolated_images(prompts, latents)

# Optionally create a GIF from the interpolated images
if interpolated_images:
    interpolated_images[0].save(
        os.path.join(OUTDIR, "interpolation.gif"),
        save_all=True,
        append_images=interpolated_images[1:],
        duration=200,
        loop=0
    )

# Clean up memory hooks after processing
hook1.remove()
hook2.remove()
hook3.remove()
pipe.transformer.cuda()  # Move transformer back to GPU
