import os
import json
import torch
from PIL import Image
import numpy as np
from packaging import version
import shutil

from accelerate import Accelerator
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from idarbdiffusion.models.unet_dr2d_condition import UNetDR2DConditionModel
from idarbdiffusion.pipelines.pipeline_idarbdiffusion import IDArbDiffusionPipeline
from idarbdiffusion.data.custom_dataset import CustomDataset
from idarbdiffusion.data.custom_mv_dataset import CustomMVDataset


def reform_image(img_nd):
    albedo, normal = img_nd[0, ...], img_nd[1, ...]
    mro = img_nd[2, ...]
    mtl, rgh = mro[:1, ...], mro[1:2, ...]
    mtl, rgh = np.repeat(mtl, 3, axis=0), np.repeat(rgh, 3, axis=0)
    img_reform = np.concatenate([albedo, normal, mtl, rgh], axis=-1)
    return img_reform.transpose(1, 2, 0)


def save_image(out, name):
    Nv = out.shape[0]
    for i in range(Nv):
        img = reform_image(out[i])
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join("./temp", f'{name}_{(i):02d}.png'))

def reform_image_sep(img_nd):
    albedo, normal = img_nd[0, ...], img_nd[1, ...]
    mro = img_nd[2, ...]
    mtl, rgh = mro[:1, ...], mro[1:2, ...]
    mtl, rgh = np.repeat(mtl, 3, axis=0), np.repeat(rgh, 3, axis=0)
    return albedo, normal, mtl, rgh

def save_images(out, name):
    """Save all four components: albedo, normal, mtl, rgh"""
    Nv = out.shape[0]
    os.makedirs("./temp_results", exist_ok=True)
    
    for i in range(Nv):
        # Extract components
        albedo, normal, mtl, rgh = reform_image_sep(out[i])
                
        # Handle unusual shapes
        def process_image(img, component_name):
            # First check if image is in channel-first format (C, H, W)
            if len(img.shape) == 3 and img.shape[0] <= 4:  # Typical channel sizes (1-4)
                img = img.transpose(1, 2, 0)  # Convert from (C,H,W) to (H,W,C)
            
            # If it's a weird shape like (1, 1, 512), reshape it to something sensible
            if len(img.shape) == 3 and img.shape[0] == 1 and img.shape[1] == 1:
                # Reshape to a square image with 3 channels
                size = int(np.sqrt(img.shape[2] // 3))
                img = img.reshape(size, size, 3)
            
            # Ensure the shape is valid for an image
            if len(img.shape) == 2:
                # Grayscale image
                pass
            elif len(img.shape) == 3 and img.shape[2] > 4:
                # Too many channels, keep only first 3
                img = img[:, :, :3]
            
            # Convert to uint8 for saving
            if img.dtype.kind == 'f':  # If float
                img = (img * 255).astype(np.uint8)
            
            return img
        
        # Process each component
        albedo = process_image(albedo, "albedo")
        normal = process_image(normal, "normal")
        mtl = process_image(mtl, "mtl")
        rgh = process_image(rgh, "rgh")
        
        # Save all components
        try:
            Image.fromarray(albedo).save(os.path.join("./temp_results", f'{name}_{(i)}_albedo.png'))
            Image.fromarray(normal).save(os.path.join("./temp_results", f'{name}_{(i)}_normal.png'))
            Image.fromarray(mtl).save(os.path.join("./temp_results", f'{name}_{(i)}_mtl.png'))
            Image.fromarray(rgh).save(os.path.join("./temp_results", f'{name}_{(i)}_rgh.png'))
        except Exception as e:
            print(f"Error saving images: {e}")
            # Fall back to a simpler format if needed
            try:
                # Try reshaping differently if the first attempt failed
                for component, name_suffix in [(albedo, "albedo"), (normal, "normal"), (mtl, "mtl"), (rgh, "rgh")]:
                    if component.size > 0:  # Ensure there's data
                        # Reshape to a simple 2D array if needed
                        flat_size = int(np.sqrt(component.size))
                        reshaped = component.flatten()[:flat_size**2].reshape((flat_size, flat_size))
                        Image.fromarray(reshaped.astype(np.uint8)).save(
                            os.path.join("./temp_results", f'{name}_{(i):02d}_{name_suffix}_fallback.png')
                        )
            except Exception as inner_e:
                print(f"Fallback also failed: {inner_e}")


def load_pipeline():
    """
    load pipeline from hub
    or load from local ckpts: pipeline = IDArbDiffusionPipeline.from_pretrained("./pipeckpts")
    """
    text_encoder = CLIPTextModel.from_pretrained("lizb6626/IDArb", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("lizb6626/IDArb", subfolder="tokenizer")
    feature_extractor = CLIPImageProcessor.from_pretrained("lizb6626/IDArb", subfolder="feature_extractor")
    vae = AutoencoderKL.from_pretrained("lizb6626/IDArb", subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained("lizb6626/IDArb", subfolder="scheduler")
    unet = UNetDR2DConditionModel.from_pretrained("lizb6626/IDArb", subfolder="unet")
    pipeline = IDArbDiffusionPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        vae=vae,
        unet=unet,
        safety_checker=None,
        scheduler=scheduler,
    )
    return pipeline

def generate_fake_transforms(root_dir):
    """generate a fake transforms.json file"""
    print(f"Generating fake transforms for {root_dir}")
    image_files = os.listdir(root_dir)
    print(f"Found {len(image_files)} images")
    transforms = {"frames":[]}
    for i in range(len(image_files)):
        transforms["frames"].append(
            {
                "file_path": f"{i}",
                "transform_matrix": "null",
            }
        )
    with open(os.path.join(root_dir, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=4)


def compute_albedo(imgs: Image.Image, input_type="single", use_cam=False) -> Image.Image:
    """
    Compute the albedo of an image.
    """
    weight_dtype = torch.float16
    pipeline = load_pipeline()
    if is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        pipeline.unet.enable_xformers_memory_efficient_attention()
        print(f'Use xformers version: {xformers_version}')

    pipeline.to("cuda")
    print("Pipeline loaded successfully")

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(os.path.join(temp_dir, f'{i}.png'))

    if input_type == 'single':
        print("Single image")
        dataset = CustomDataset(
            root_dir=temp_dir,
        )
    else:
        print("Multiple images")
        os.makedirs("./temp/mv", exist_ok=True)
        for i, img in enumerate(imgs):
            img.save(os.path.join("./temp/mv", f'{i}.png'))
        generate_fake_transforms("./temp/mv")

        dataset = CustomMVDataset(
            root_dir=temp_dir,
            num_views=len(imgs),
            use_cam=use_cam,
        )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    Nd = 3

    for i, batch in enumerate(dataloader):

        imgs_in, imgs_mask, task_ids = batch['imgs_in'], batch['imgs_mask'], batch['task_ids']
        cam_pose = batch['pose']
        imgs_name = batch['data_name']

        imgs_in = imgs_in.to(weight_dtype).to("cuda")
        cam_pose = cam_pose.to(weight_dtype).to("cuda")

        B, Nv, _, H, W = imgs_in.shape


        imgs_in, imgs_mask, task_ids = imgs_in.flatten(0,1), imgs_mask.flatten(0,1), task_ids.flatten(0,2)

        with torch.autocast("cuda"):
            out = pipeline(
                    imgs_in,
                    task_ids,
                    num_views=Nv,
                    cam_pose=cam_pose,
                    height=H, width=W,
                    # generator=generator,
                    guidance_scale=1.0,
                    output_type='pt',
                    num_images_per_prompt=1,
                    eta=1.0,
                ).images

            out = out.view(B, Nv, Nd, *out.shape[1:])
            out = out.detach().cpu().numpy()
            print(out.shape)
            for i in range(B):
                save_images(out[i], imgs_name[i])
            return out


if __name__ == "__main__":
    input_dir = "./temp"
    imgs = [Image.open(os.path.join(input_dir, f'{i}.png')) for i in range(4)]
    compute_albedo(imgs, input_type="multi", use_cam=False)