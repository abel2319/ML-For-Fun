# Import needed libraries
import streamlit as st
import torchvision
from medigan import Generators
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

# Define the GAN models available in the app
models = {
    "Breast Calcification": "00001_DCGAN_MMG_CALC_ROI",
    "Breast Mass": "00002_DCGAN_MMG_MASS_ROI",
    "Breast Density Transfer": "00003_CYCLEGAN_MMG_DENSITY_FULL",
    "Breast Mass with Mask": "00004_PIX2PIX_MMG_MASSES_W_MASKS",
    "Breast Mass (BCDR 1)": "00005_DCGAN_MMG_MASS_ROI",
    "Breast Mass (BCDR 2)": "00006_WGANGP_MMG_MASS_ROI",
    "Brain Tumors on Flair, T1, T1c, T2 with Masks": "00007_INPAINT_BRAIN_MRI",
    "Breast Mass (Mal/Benign - CBIS-DDSM)": "00008_C-DCGAN_MMG_MASSES",
    "Polyp with Mask (PGGAN)": "00009_PGGAN_POLYP_PATCHES_W_MASKS",
    "Polyp with Mask (FastGAN)": "00010_FASTGAN_POLYP_PATCHES_W_MASKS",
    "Polyp with Mask (SinGAN)": "00011_SINGAN_POLYP_PATCHES_W_MASKS",
    "Breast Mass (Mal/Benign - BCDR)": "00012_C-DCGAN_MMG_MASSES",
    "Breast Density Transfer MLO (OPTIMAM)": "00013_CYCLEGAN_MMG_DENSITY_OPTIMAM_MLO",
    "Breast Density Transfer CC (OPTIMAM)": "00014_CYCLEGAN_MMG_DENSITY_OPTIMAM_CC",
    "Breast Density Transfer MLO (CSAW)": "00015_CYCLEGAN_MMG_DENSITY_CSAW_MLO",
    "Breast Density Transfer CC (CSAW)": "00016_CYCLEGAN_MMG_DENSITY_CSAW_CC",
    "Lung Nodules (DCGAN)": "00017_DCGAN_XRAY_LUNG_NODULES",
    "Lung Nodules (WGAN-GP)": "00018_WGANGP_XRAY_LUNG_NODULES",
    "Chest Xray Images (1)": "00019_PGGAN_CHEST_XRAY",
    "Chest Xray Images (2)": "00020_PGGAN_CHEST_XRAY",
    "Brain T1-T2 MRI Modality Transfer": "00021_CYCLEGAN_BRAIN_MRI_T1_T2",
    "Cardiac MRI Age Transfer": "00022_WGAN_CARDIAC_AGING",
    "Breast DCE-MRI Contrast Injection": "00023_PIX2PIXHD_BREAST_DCEMRI"
}


def main():
    st.title("MediGen")
    st.write("Generate synthetic medical images using pre-trained GAN models.")
    # Add dropdown widget for model selection to the sidebar
    model_id = st.sidebar.selectbox("Select Model ID", models.keys())

    # Add number image selector to the sidebar
    num_images = st.sidebar.number_input(
        "Number of Images", min_value=1, max_value=7, value=1, step=1
    )


    # Add generate button to the sidebar
    if st.sidebar.button("Generate Images"):
        generate_images(num_images, models[model_id])


def torch_images(num_images, model_id):
    generators = Generators()
    dataloader = generators.get_as_torch_dataloader(
        model_id=model_id,
        install_dependencies=True,
        num_samples=num_images,
        prefetch_factor=None,
    )

    images = []
    for batch_idx, data_dict in enumerate(dataloader):
        image_list = []
        for i in data_dict:
            if "sample" in i:
                sample = data_dict.get("sample")
                if sample.dim() == 4:
                    sample = sample.squeeze(0).permute(2, 0, 1)

                sample = to_pil_image(sample).convert("RGB")
                # Convert the image to a PyTorch tensor
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                )

                # Apply the transform to your PIL image
                sample = transform(sample)
                image_list.append(sample)

            # Preprocess the mask
            if "mask" in i:
                mask = data_dict.get("mask")
                if mask.dim() == 4:
                    mask = mask.squeeze(0).permute(2, 0, 1)
                mask = to_pil_image(mask).convert("RGB")
                mask = transform(mask)
                image_list.append(mask)

        # Organize the grid to have 'sample' images per row
        Grid = make_grid(image_list, nrow=2)

        # Change Grid tensor to be a consistent shape
        # The Grid tensor has shape [1, 128, 128, 1] in some models
        if Grid.dim() == 4:
            # Remove the singleton batch dimension
            Grid = Grid.squeeze(0)
            if Grid.size(-1) == 1:
                # Remove the singleton channel dimension (assuming grayscale)
                Grid = Grid.squeeze(-1)
            else:
                raise ValueError("Expected a single channel (grayscale) image.")

        # Convert the tensor grid to a PIL Image for display
        img = torchvision.transforms.ToPILImage()(Grid)
        images.append(img)
    return images


def generate_images(num_images, model_id):
    st.subheader("Generated Images:")
    images = torch_images(num_images, model_id)
    for i in range(len(images)):
        st.image(
            images[i],
            caption=f"Generated Image {i+1} (Model ID: {model_id})",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
