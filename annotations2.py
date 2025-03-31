# file: process_freiburg_dataset.py
# This file is used to compute the pseudo-groundtruth to create the dataset for training dust3r.

import sys
sys.path.append('./mast3r')
sys.path.append('./dust3r')

import os
import json
import shutil
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
import dust3r.datasets.utils.cropping as cropping


def extract_timestamp(filename):
    parts = filename.rstrip(".png").split("_")[-2:]
    if len(parts) == 2:
        sec, nsec = parts
        return float(f"{sec}.{nsec}")
    return 0.0

def make_intrinsics_np(focal, pp):
    """Creates a 3x3 intrinsic camera matrix from focal length and principal point values"""
    if focal.ndim == 0:
        fx = fy = float(focal)
    else:
        fx, fy = focal
    cx, cy = pp
    return np.array([[float(fx), 0, cx], [0, float(fy), cy], [0, 0, 1]], dtype=np.float32)

def load_model(model_path, device):
    return AsymmetricMASt3R.from_pretrained(model_path).to(device)

def save_progress(dataset_entries, output_json, count):
    """Saves dataset entries to a JSON file and prints a progress message"""
    
    with open(output_json, "w") as f:
        json.dump(dataset_entries, f, indent=4)
    print(f"Saved progress after {count} images.")

def reset_cache(cache_path="/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/cache1"):
    """Resets the cache by deleting the existing cache folder and creating a new one"""

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

def crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng=None, aug_crop=0, info=None):
    """
    Crops and resizes an image and depthmap if necessary, adjusting the intrinsics based on the given resolution and optional augmentations.
    Taken from the BaseStereoVision class from dust3r to insure images are in correct dimensions before saving them to the dataset.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    if min_margin_x < W // 5 or min_margin_y < H // 5:
        print(f"Principal point too close to image border for view={info}")

    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    W, H = image.size
    if resolution[0] >= resolution[1]:
        if H > 1.1 * W:
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            if rng.integers(2):
                resolution = resolution[::-1]

    target_resolution = np.array(resolution)
    if aug_crop > 1:
        target_resolution += rng.integers(0, aug_crop)

    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    return image, depthmap, intrinsics2

def process_image_pair(img1, img2, ir_img1, ir_img2, model, device, shared_intrinsics, image_size, output_depth_dir, resized_img_dir):
    """
    Processes a pair of images by resizing, aligning, extracting depthmaps, pose information and intrinsics and saving the results
    in a JSON format.
    """
    os.makedirs(resized_img_dir, exist_ok=True)

    # Resize and save the images to resized_img_dir
    img1_resized_path = os.path.join(resized_img_dir, os.path.basename(img1))
    img2_resized_path = os.path.join(resized_img_dir, os.path.basename(img2))

    Image.open(img1).convert("RGB").resize((image_size, image_size), resample=Image.Resampling.LANCZOS).save(img1_resized_path)
    Image.open(img2).convert("RGB").resize((image_size, image_size), resample=Image.Resampling.LANCZOS).save(img2_resized_path)

    # Reload resized images and pass to sparse alignment
    filelist = [img1_resized_path, img2_resized_path]
    imgs = load_images(filelist, size=image_size, verbose=True, square_ok=True)
    pairs = make_pairs(imgs, scene_graph='complete', symmetrize=True)

    scene = sparse_global_alignment(
        imgs=filelist,
        pairs_in=pairs,
        cache_path="/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/cache1",
        model=model,
        lr1=0.2, niter1=1000,
        lr2=0.02, niter2=10,
        device=device,
        opt_depth=True,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=5.0
    )

    pts3d, depthmaps_dense, confs = scene.get_dense_pts3d(clean_depth=True)

    focal0, focal1 = scene.get_focals()[:2]
    pp0, pp1 = scene.get_principal_points()[:2]
    K0 = make_intrinsics_np(focal0.detach().cpu().numpy(), pp0.detach().cpu().numpy()).tolist()
    K1 = make_intrinsics_np(focal1.detach().cpu().numpy(), pp1.detach().cpu().numpy()).tolist()
    pose0 = scene.get_im_poses()[0].cpu().numpy().tolist()
    pose1 = scene.get_im_poses()[1].cpu().numpy().tolist()
    depthmap0 = depthmaps_dense[0].cpu().numpy()
    depthmap1 = depthmaps_dense[1].cpu().numpy()

    depth_filename1 = os.path.join(output_depth_dir, f"depth_{os.path.basename(img1)}.npy")
    depth_filename2 = os.path.join(output_depth_dir, f"depth_{os.path.basename(img2)}.npy")

    s01, s02 = confs[0].shape
    s11, s12 = confs[1].shape

    image0 = np.array(Image.open(img1).convert("RGB").resize((s01, s02), resample=Image.Resampling.LANCZOS))
    image1 = np.array(Image.open(img2).convert("RGB").resize((s11, s12), resample=Image.Resampling.LANCZOS))

    rng = np.random.default_rng()
    try:
        _ = crop_resize_if_necessary(image0, depthmap0.reshape(*confs[0].shape), np.array(K0), [image_size, image_size], rng=rng, info=img1)
        _ = crop_resize_if_necessary(image1, depthmap1.reshape(*confs[1].shape), np.array(K1), [image_size, image_size], rng=rng, info=img2)
    except Exception as e:
        raise RuntimeError(f"Image pair failed cropping check: {img1}, {img2}\n{e}")
        # continue

    np.save(depth_filename1, depthmap0)
    np.save(depth_filename2, depthmap1)

    entry1 = {
        "rgb_path": img1_resized_path,
        "ir_path": ir_img1,
        "intrinsics": K0,
        "extrinsics": pose0,
        "depth_map_path": depth_filename1,
        "shape": list(map(int, confs[0].shape)),
    }
    entry2 = {
        "rgb_path": img2_resized_path,
        "ir_path": ir_img2,
        "intrinsics": K1,
        "extrinsics": pose1,
        "depth_map_path": depth_filename2,
        "shape": list(map(int, confs[1].shape)),
    }

    return entry1, entry2


def process_freiburg_dataset(image_size = 224, output_json="dataset_info_o1.json", output_depth_dir="depth_maps_t1", 
                            resized_img_dir="resized_dir", n_save_interval=10):
    
    """Processes the Freiburg dataset by extracting image pairs, generating depthmaps, and saving metadata to a JSON file at regular intervals."""
    
    root_dir = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/freiburg_dataset/train"
    shared_intrinsics = True
    model_path = "./naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    os.makedirs(output_depth_dir, exist_ok=True)
    reset_cache()

    device = 'cuda'
    model = load_model(model_path, device)

    dataset_entries = []
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            dataset_entries = json.load(f)

    # processed_paths = set(entry["rgb_path"] for entry in dataset_entries)
    processed_paths = set(os.path.basename(entry["rgb_path"]) for entry in dataset_entries)

    image_count = 0

    for seq in os.listdir(root_dir):
        seq_path = os.path.join(root_dir, seq)
        if not os.path.isdir(seq_path):
            continue

        for sub_seq in os.listdir(seq_path):
            sub_seq_path = os.path.join(seq_path, sub_seq)
            if not os.path.isdir(sub_seq_path):
                continue

            rgb_path = os.path.join(sub_seq_path, "fl_rgb")
            ir_path = os.path.join(sub_seq_path, "fl_ir_aligned")

            if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
                continue

            rgb_images = sorted([f for f in os.listdir(rgb_path) if f.endswith(".png")], key=extract_timestamp)

            for i in range(0, len(rgb_images) - 1, 2):
                img1 = os.path.join(rgb_path, rgb_images[i])
                img2 = os.path.join(rgb_path, rgb_images[i + 1])
                ir_img1 = os.path.join(ir_path, rgb_images[i].replace("fl_rgb", "fl_ir_aligned"))
                ir_img2 = os.path.join(ir_path, rgb_images[i + 1].replace("fl_rgb", "fl_ir_aligned"))

                if os.path.basename(img1) in processed_paths and os.path.basename(img2) in processed_paths:
                    print(f"Skipping {img1} and {img2}, already processed.")
                    continue

                try:
                    entry1, entry2 = process_image_pair(img1, img2, ir_img1, ir_img2, model, device,
                                                    shared_intrinsics, image_size, output_depth_dir, resized_img_dir)
                except Exception as e:
                    print(f"Failed to process pair: {img1}, {img2}\n{e}")
                    continue

                dataset_entries.extend([entry1, entry2])
                processed_paths.update([img1, img2])
                image_count += 2

                if image_count % n_save_interval == 0:
                    save_progress(dataset_entries, output_json, image_count)
                    reset_cache()

    save_progress(dataset_entries, output_json, image_count)
    print(f"Final dataset processing complete. Saved metadata in {output_json}.")

if __name__ == "__main__":
    output_json = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/dataset_v1_224.json"
    output_depth_dir = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/dataset_v1_224_depth"
    resized_img_dir = "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/dataset_v1_224_imgs"
    process_freiburg_dataset(output_json=output_json, output_depth_dir=output_depth_dir, resized_img_dir=resized_img_dir, n_save_interval=2)
