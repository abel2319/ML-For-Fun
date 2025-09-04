# MediGen: Synthetic Medical Image Generator

**MediGen** is a Streamlit-based application for generating **synthetic medical images** using various pre-trained GAN models provided by [MediGAN](https://github.com/DIAGNijmegen/medigan). This app allows users to explore and visualize diverse medical image modalities such as breast mammography, brain MRI, chest X-rays, and more.

---

## 🚀 Features

-  Easy-to-use Streamlit UI
-  20+ pre-trained GAN models for different medical image types
-  Supports image generation with masks (for segmentation) where applicable
-  Automatically handles dependencies and model fetching via `medigan`

---

## Supported Models

Examples include:
- Breast Calcification (DCGAN)
- Breast Density Transfer (CycleGAN)
- Brain Tumor MRI with Masks (Inpainting)
- Polyp Generation (PGGAN, FastGAN, SinGAN)
- Chest X-ray Generation (PGGAN)
- Cardiac MRI Age Transfer (WGAN)

> A full list of models is included in the app sidebar.

---

## 📦 Installation

1. **Clone this repository**

```bash
git clone https://github.com/your-username/medigen-app.git
cd medigen-app
```
2. Install dependencies

I recommend using a virtual environment.
```bash
pip install streamlit torch torchvision medigan
```

3. **Run the app**
```bash
streamlit run app.py
```
---

## 🧠 How It Works
- The app uses medigan's Generators interface to fetch and use pre-trained models.
- The user selects a model and number of images from the sidebar.
- The selected model generates synthetic images (and masks if available).
- Images are processed into a grid and displayed within the Streamlit interface.