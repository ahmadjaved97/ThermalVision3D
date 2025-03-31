import sys
import json
import os
import cv2
import numpy as np
import torch

sys.path.append("/home/s63ajave_hpc/dust3r")

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from rich import print


def enhance_thermal_image(image):
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
    if use_rgb:
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
    rgb_path = ir_path.replace("fl_ir_aligned", "fl_rgb")
    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    return rgb_img


def save_depth_and_overlay(depth, overlay_base_img, filename_prefix):
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(f"{filename_prefix}_depth.png", depth_colored)

    overlay = cv2.addWeighted(depth_colored, 0.6, cv2.cvtColor(overlay_base_img, cv2.COLOR_RGB2BGR), 0.4, 0)
    cv2.imwrite(f"{filename_prefix}_overlay.png", overlay)


def save_groundtruth_npy_image(npy_path, output_prefix, resolution=224):
    if os.path.exists(npy_path):
        gt = np.load(npy_path)
        gt = gt.reshape((resolution, resolution))
        gt_norm = cv2.normalize(gt, None, 0, 255, cv2.NORM_MINMAX)
        gt_uint8 = gt_norm.astype(np.uint8)
        gt_colored = cv2.applyColorMap(gt_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(f"{output_prefix}_gt.png", gt_colored)


def run_depth_inference_with_paths(img1_path, img2_path, model, output_prefix="img", resolution=224, use_enhance=True):
    os.makedirs("temp_inputs", exist_ok=True)
    temp1_path, _ = preprocess_ir_and_save(img1_path, "temp_inputs/temp1.png", resolution, use_enhance, use_rgb=False)
    temp2_path, _ = preprocess_ir_and_save(img2_path, "temp_inputs/temp2.png", resolution, use_enhance, use_rgb=False)

    temp_images = load_images([temp1_path, temp2_path], size=224)
    rgb1 = load_rgb_image_from_ir_path(img1_path, resolution)
    rgb2 = load_rgb_image_from_ir_path(img2_path, resolution)

    output = inference([tuple(temp_images)], model, device=device, batch_size=1)

    depth1 = output['pred1']['pts3d'][..., 2].squeeze(0).cpu().numpy()
    depth2 = output['pred2']['pts3d_in_other_view'][..., 2].squeeze(0).cpu().numpy()

    save_depth_and_overlay(depth1, rgb1, f"{output_prefix}1")
    save_depth_and_overlay(depth2, rgb2, f"{output_prefix}2")

    print(f"✅ Inference done for {output_prefix}")


# === Run with JSON Input ===
if __name__ == "__main__":
    json_file = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/dataset_v1_224_test.json"  # <-- replace with actual path
    model_path = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/checkpoints/dust3r_freiburg_224_thermal8/checkpoint-best.pth"
    resolution = 224
    use_enhance = False

    with open(json_file, 'r') as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    idx = 0
    for i in range(0, len(data) - 1, 2):
        ir_path1 = data[i]['ir_path']
        ir_path2 = data[i+1]['ir_path']
        gt_path1 = data[i].get('depth_map_path')
        gt_path2 = data[i+1].get('depth_map_path')

        output_prefix = f"pair_{i}_{i+1}"
        run_depth_inference_with_paths(ir_path1, ir_path2, model, output_prefix, resolution, use_enhance)

        if gt_path1:
            save_groundtruth_npy_image(gt_path1, f"{output_prefix}1", resolution)
        if gt_path2:
            save_groundtruth_npy_image(gt_path2, f"{output_prefix}2", resolution)
        
        idx += 1
        if idx > 10:
            break

