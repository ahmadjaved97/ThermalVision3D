# file: test_models_ais.py
# This file is used to get the depthmaps for the images present in the AIS(FLIR_BOSON) dataset and visualize them.

import sys
sys.path.append("/home/s63ajave_hpc/dust3r")

import sys
import os
import cv2
import numpy as np
import torch
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from rich import print
import re

def natural_sort(files):
    # Sort files using natural sorting order
    return sorted(files, key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])


def enhance_thermal_image(image):
    """Enhances thermal images using CLAHE, sharpening, and Gaussian blur techniques."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_planes = list(cv2.split(image_lab))
    lab_planes[0] = clahe.apply(lab_planes[0])
    image_clahe = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(image_clahe, cv2.COLOR_LAB2RGB)

    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    final = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return final


def preprocess_ir_and_save(image_path, save_path, resolution=224, use_enhance=True, use_rgb=False):
    """Preprocesses and saves an image (either RGB or grayscale) with optional enhancement"""
    if use_rgb == True:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image] * 3, axis=-1)

    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    if use_enhance:
        image = enhance_thermal_image(image)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) 
    return save_path, image


def load_rgb_image_from_ir_path(ir_path, resolution=224):
    """Load RGB Image by modifying the Thermal image path"""
    rgb_path = ir_path.replace("t", "c")
    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    return rgb_img

def load_model(model_path, device='cuda'):
    """Load model from model path"""
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    return model

def save_depth_and_overlay(depth, overlay_base_img, filename_prefix):
    """
    Function to normalize and color the depth map, then overlays it onto the thermal image, saving both the depth map and the overlayed image
    """
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(f"{filename_prefix}_depth.png", depth_colored)

    # overlay = cv2.addWeighted(depth_colored, 0.6, cv2.cvtColor(overlay_base_img, cv2.COLOR_RGB2BGR), 0.4, 0)
    # cv2.imwrite(f"{filename_prefix}_overlay.png", overlay)
    # Overlay depth map on the thermal image (grayscale)
    thermal_colored = cv2.cvtColor(overlay_base_img, cv2.COLOR_GRAY2BGR)  # Convert thermal to 3-channel image
    overlay = cv2.addWeighted(depth_colored, 0.4, thermal_colored, 0.6, 0)  # Overlay the depth map on the thermal image
    cv2.imwrite(f"{filename_prefix}_overlay.png", overlay)


def run_depth_inference_with_paths(img1_path, img2_path, model, output_prefix="img", resolution=224, use_enhance=True, device='cuda'):
    """Function to perform depth inference on two thermal images"""

    os.makedirs("temp_inputs", exist_ok=True)
    temp1_path, _ = preprocess_ir_and_save(img1_path, "temp_inputs/temp1.png", resolution, use_enhance, use_rgb=False)
    temp2_path, _ = preprocess_ir_and_save(img2_path, "temp_inputs/temp2.png", resolution, use_enhance, use_rgb=False)
    print(img1_path)
    print(img2_path)

    temp_paths = [temp1_path, temp2_path]
    temp_images = load_images(temp_paths, size=224)

    # rgb1 = load_rgb_image_from_ir_path(img1_path, resolution)
    # rgb2 = load_rgb_image_from_ir_path(img2_path, resolution)

    # Load the thermal images (grayscale)
    thermal1 = cv2.imread(temp1_path, cv2.IMREAD_GRAYSCALE)
    thermal2 = cv2.imread(temp2_path, cv2.IMREAD_GRAYSCALE)

    output = inference([tuple(temp_images)], model, device=device, batch_size=1)

    depth1 = output['pred1']['pts3d'][..., 2].squeeze(0).cpu().numpy()
    depth2 = output['pred2']['pts3d_in_other_view'][..., 2].squeeze(0).cpu().numpy()
    
    save_depth_and_overlay(depth1, thermal1, f"{output_prefix}1")
    save_depth_and_overlay(depth2, thermal2, f"{output_prefix}2")

    print(f"Inference done. Overlay saved on RGB images at '{output_prefix}1_*.png' and '{output_prefix}2_*.png'")


def process_folder(folder_path, model, output_prefix="thermal_pair", resolution=224, use_enhance=False, device='cuda'):
    # Get all thermal image paths
    thermal_images = [f for f in os.listdir(folder_path) if "t" in f]
    thermal_images = natural_sort(thermal_images)

    for i in range(0,len(thermal_images) - 1,2):
        img1_path = os.path.join(folder_path, thermal_images[i])
        img2_path = os.path.join(folder_path, thermal_images[i + 1])
        
        run_depth_inference_with_paths(img1_path, img2_path, model, 
                                       output_prefix=f"{output_prefix}_{i}", resolution=resolution, 
                                       use_enhance=use_enhance, device=device)


if __name__ == "__main__":
    folder_path = "./flir_boson/s1"
    model_path = "./dust3r_freiburg_224_thermal8/checkpoint-best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    
    process_folder(folder_path, model, output_prefix="thermal_pair", use_enhance=False, device=device)

