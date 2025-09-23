# Stable Diffusion GUI

This is a simple web-based GUI for generating images using the **Stable Diffusion v1-4** model. It uses **Streamlit** for the frontend and the **Diffusers** library from Hugging Face for image generation.

Users can input a text prompt and select image generation parameters such as:
- Number of images
- Guidance scale
- Inference steps
- Artistic style (e.g., anime, photo, video game, watercolor)

## Features

- Text-to-image generation with Stable Diffusion
- Multiple image generation at once (up to 10)
- Adjustable inference settings
- Style presets for artistic customization
- GPU acceleration (if available)

## Requirements

Make sure you have the following Python packages installed:

- `diffusers`
- `torch`
- `streamlit`

You can install them via pip:

```bash
pip install diffusers torch streamlit
```

## Running the App

To start the app, run the following command in your terminal:

```bash
streamlit run app.py
```

## Notes

* GPU is recommended for faster image generation.
* The app uses the pretrained model CompVis/stable-diffusion-v1-4.
* Internet connection is required for downloading the model the first time.