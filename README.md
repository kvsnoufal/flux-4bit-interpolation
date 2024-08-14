# Flux.1 with 4 bit Quantization for prompt interpolation

This is a fork of the [github repo] (https://github.com/HighCWu/flux-4bit)

To use the Flux.1 dev 4-bit quantized for image generation refer the [original repo](https://github.com/HighCWu/flux-4bit)
<div align = center>

[![Badge Model]][Model]   
[![Badge Colab]][Colab]

<!---------------------------------------------------------------------------->

[Model]: https://huggingface.co/HighCWu/FLUX.1-dev-4bit
[Colab]: https://colab.research.google.com/github/HighCWu/flux-4bit/blob/main/colab_t4.ipynb


<!---------------------------------[ Badges ]---------------------------------->

[Badge Model]: https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg
[Badge Colab]: https://colab.research.google.com/assets/colab-badge.svg

<!---------------------------------------------------------------------------->
</div>

This project runs prompt interpolation between multiple prompt encodings and generates a gif as output.

## How to use:

1. clone the repo:
    ```sh
    git clone https://github.com/kvsnoufal/flux-4bit-interpolation
    cd flux-4bit-interpolation
    ```

2. install requirements:
    ```sh
    pip install -r requirements.txt
    ```

3. run `run_interpolation.py`:
    ```sh
    python run_interpolation.py
    ```

## Features
- **Text-to-Image Generation**: Uses AI models to convert text descriptions into images.
- **Spherical Linear Interpolation (Slerp)**: Interpolates between image embeddings to create smooth transitions between different stages.
- **Customizable Prompts**: Set prompts to interpolate
- **GIF Generation**: Optionally combines the interpolated images into a GIF for easy visualization.

## Project Structure

- **`model.py`: Contains the custom model definitions for the T5 encoder and Flux Transformer.
- **`run_interpolation.py`**: Main script for generating the interpolated images.

## Acknowledgments

- [HighCWu/FLUX.1-dev-4bit](https://huggingface.co/HighCWu/FLUX.1-dev-4bit) for the quantized models.
- [Hugginface model repo](https://huggingface.co/HighCWu/FLUX.1-dev-4bit)
- [Article on Latent Space interpolation](https://huggingface.co/learn/cookbook/en/stable_diffusion_interpolation)

