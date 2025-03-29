import sys
sys.path.append("/home/s63ajave_hpc/dust3r")

import os
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from skimage.metrics import structural_similarity as ssim
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


def preprocess_and_save(image_path, save_path, resolution=224, use_enhance=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.stack([image] * 3, axis=-1)
    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    if use_enhance:
        image = enhance_thermal_image(image)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return save_path


def compute_depth_metrics(pred, gt):
    mask = (gt > 0)
    pred = pred[mask]
    gt = gt[mask]
    thresh = np.maximum(gt / pred, pred / gt)
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min())
    pred_img = (pred_norm * 255).astype(np.uint8)
    gt_img = (gt_norm * 255).astype(np.uint8)
    ssim_score = ssim(gt_img, pred_img)

    return {
        "RMSE": np.sqrt(np.mean((pred - gt) ** 2)),
        "AbsRel": np.mean(np.abs(pred - gt) / gt),
        "SSIM": ssim_score,
        "Acc<1.25": (thresh < 1.25).mean(),
        "Acc<1.25^2": (thresh < 1.25 ** 2).mean(),
        "Acc<1.25^3": (thresh < 1.25 ** 3).mean()
    }


def main(metadata_path, model_path, resolution=224, use_rgb=False, use_enhance=False):

    if use_rgb and use_enhance:
        print("[yellow]⚠️ 'use_enhance' is ignored because 'use_rgb' is True.[/yellow]")
        use_enhance = False

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # metadata = metadata[int(0.95 * len(metadata)):]  # test split
    pairs = [(i, i + 1) for i in range(0, len(metadata), 2) if i + 1 < len(metadata)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    os.makedirs("temp_inputs", exist_ok=True)
    os.makedirs("depth_outputs", exist_ok=True)

    all_metrics = {k: [] for k in ["RMSE", "AbsRel", "SSIM", "Acc<1.25", "Acc<1.25^2", "Acc<1.25^3"]}

    for idx, (i1, i2) in enumerate(tqdm(pairs)):
        meta1, meta2 = metadata[i1], metadata[i2]
        img1_path = meta1["rgb_path"] if use_rgb else meta1["ir_path"]
        img2_path = meta2["rgb_path"] if use_rgb else meta2["ir_path"]
        gt_path = meta1["depth_map_path"]

        temp1_path = preprocess_and_save(img1_path, f"temp_inputs/temp1_{idx:02d}.png", resolution, use_enhance)
        temp2_path = preprocess_and_save(img2_path, f"temp_inputs/temp2_{idx:02d}.png", resolution, use_enhance)

        images = load_images([temp1_path, temp2_path], size=resolution)
        output = inference([tuple(images)], model, device=device, batch_size=1, verbose=None)

        pred_depth = output['pred1']['pts3d'][..., 2].squeeze(0).cpu().numpy()
        gt_depth = np.load(gt_path).reshape(resolution, resolution).astype(np.float32)

        if pred_depth.shape != gt_depth.shape:
            gt_depth = cv2.resize(gt_depth, (pred_depth.shape[1], pred_depth.shape[0]))

        metrics = compute_depth_metrics(pred_depth, gt_depth)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        pred_depth_2 = output['pred2']['pts3d_in_other_view'][..., 2].squeeze(0).cpu().numpy()
        gt_depth_2 = np.load(meta2["depth_map_path"]).reshape(resolution, resolution).astype(np.float32)

        if pred_depth_2.shape != gt_depth_2.shape:
            gt_depth_2 = cv2.resize(gt_depth_2, (pred_depth_2.shape[1], pred_depth_2.shape[0]))

        metrics_2 = compute_depth_metrics(pred_depth_2, gt_depth_2)
        for k, v in metrics_2.items():
            all_metrics[k].append(v)

        if idx < 10:
            save_dir = "depth_outputs"
            os.makedirs(save_dir, exist_ok=True)

            # --- Save predicted depth ---
            depth_norm = cv2.normalize(pred_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, f"pred_depth_{idx:02d}.png"), depth_colored)

            # --- Save ground-truth depth ---
            gt_norm = cv2.normalize(gt_depth, None, 0, 255, cv2.NORM_MINMAX)
            gt_colored = cv2.applyColorMap(gt_norm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, f"gt_depth_{idx:02d}.png"), gt_colored)

    print("\n✅ [Final Average Metrics]")
    for k in all_metrics:
        print(f"{k}: {np.mean(all_metrics[k]):.4f}")


if __name__ == "__main__":
    metadata_path = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/dataset_v1_224_test.json"
    model_path = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/checkpoints/dust3r_freiburg_224_thermal3/checkpoint-best.pth"
    # model_path = "/home/s63ajave_hpc/ThermalVision3D/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    resolution = 224
    use_rgb = False
    use_enhance = True
    main(metadata_path, model_path, resolution, use_rgb, use_enhance)
