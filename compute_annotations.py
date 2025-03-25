import sys
sys.path.append('../mast3r')  # Add MASt3R to system path
sys.path.append('../dust3r')  # Add DUSt3R to system path

import os
import json
import torch
import numpy as np
import copy
from tqdm import tqdm  # ✅ Added progress bar
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs



# Define dataset root
root_dir = "/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg/train"
output_json = "dataset_info_train.json"
output_depth_dir = "depth_maps_train"
shared_intrinsics = True
n_save_interval = 10  # Save JSON every `n` images processed

# Create output directory for depth maps
os.makedirs(output_depth_dir, exist_ok=True)

# Load MASt3R model
device = 'cuda'
model_name = "../naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

# Load existing JSON (if any)
if os.path.exists(output_json):
    with open(output_json, "r") as f:
        dataset_entries = json.load(f)
else:
    dataset_entries = []

# ✅ Store processed image paths for quick lookup
processed_paths = set(entry["rgb_path"] for entry in dataset_entries)

# Iterate through each sequence
image_count = 0  # Track number of processed images
total_images = sum(len(os.listdir(os.path.join(root_dir, seq, sub_seq, "fl_rgb"))) // 2 
                   for seq in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, seq))
                   for sub_seq in os.listdir(os.path.join(root_dir, seq)) 
                   if os.path.isdir(os.path.join(root_dir, seq, sub_seq)))

with tqdm(total=total_images, desc="Processing Images") as pbar:  # ✅ Added progress bar
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

            # Get all RGB images
            rgb_images = sorted([f for f in os.listdir(rgb_path) if f.endswith(".png")])

            # Process images in pairs
            for i in range(0, len(rgb_images) - 1, 2):  # **Skip one image each iteration**
                img1 = os.path.join(rgb_path, rgb_images[i])
                img2 = os.path.join(rgb_path, rgb_images[i + 1])
                ir_img1 = os.path.join(ir_path, rgb_images[i].replace("fl_rgb", "fl_ir_aligned"))
                ir_img2 = os.path.join(ir_path, rgb_images[i + 1].replace("fl_rgb", "fl_ir_aligned"))

                # ✅ Skip if already processed
                if img1 in processed_paths and img2 in processed_paths:
                    print(f"🔄 Skipping {img1} and {img2}, already processed.")
                    pbar.update(1)  # Update progress bar
                    continue

                # Load images
                imgs = load_images([img1, img2], size=224, verbose=True)

                if len(imgs) == 1:
                    imgs = [imgs[0], copy.deepcopy(imgs[0])]
                    imgs[1]['idx'] = 1

                # Generate scene graph-based image pairs
                pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
                # Extract `true_shape` from pairs
                true_shape1 = list(map(int, pairs[0][0]["true_shape"].flatten()))  # Image 1 shape
                true_shape2 = list(map(int, pairs[0][1]["true_shape"].flatten()))  # Image 2 shape

                # Run sparse global alignment
                cache_dir = "./cache"
                os.makedirs(cache_dir, exist_ok=True)
                scene, _ = sparse_global_alignment([img1, img2], pairs, cache_dir, model, device=device, shared_intrinsics=shared_intrinsics)

                # Extract Intrinsics
                focal_lengths = scene.get_focals().cpu().numpy()
                principal_points = scene.get_principal_points().cpu().numpy()

                K1 = [[float(focal_lengths[0]), 0, float(principal_points[0][0])],
                      [0, float(focal_lengths[0]), float(principal_points[0][1])],
                      [0, 0, 1]]

                K2 = [[float(focal_lengths[1]), 0, float(principal_points[1][0])],
                      [0, float(focal_lengths[1]), float(principal_points[1][1])],
                      [0, 0, 1]]
                print(K1)
                print(K2)

                # Extract Extrinsics (For Both Images)
                poses = scene.get_im_poses().cpu().numpy()
                R1, t1 = poses[0][:3, :3].tolist(), poses[0][:3, 3].tolist()
                R2, t2 = poses[1][:3, :3].tolist(), poses[1][:3, 3].tolist()

                # Get Depth Maps
                _, dense_depthmaps, _ = scene.get_dense_pts3d()
                depthmap1 = dense_depthmaps[0].cpu().numpy()
                depthmap2 = dense_depthmaps[1].cpu().numpy()

                # Save depth maps
                depth_filename1 = os.path.join(output_depth_dir, f"depth_{os.path.basename(img1)}.npy")
                depth_filename2 = os.path.join(output_depth_dir, f"depth_{os.path.basename(img2)}.npy")
                np.save(depth_filename1, depthmap1)
                np.save(depth_filename2, depthmap2)

                # ✅ Save metadata to JSON
                dataset_entries.append({
                    "rgb_path": img1,
                    "ir_path": ir_img1,
                    "intrinsics": K1,
                    "extrinsics": {"rotation": R1, "translation": t1},
                    "depth_map_path": depth_filename1,
                    "original_shape": true_shape1,  # Convert to standard Python int
                    "modified_shape": list(map(int, depthmap1.shape))  # Convert shape to standard Python int
                })

                dataset_entries.append({
                    "rgb_path": img2,
                    "ir_path": ir_img2,
                    "intrinsics": K2,
                    "extrinsics": {"rotation": R2, "translation": t2},
                    "depth_map_path": depth_filename2,
                    "original_shape": true_shape2,  # Convert to standard Python int
                    "modified_shape": list(map(int, depthmap2.shape))  # Convert shape to standard Python int
                })

                image_count += 2  # Since we process two images per iteration
                processed_paths.update([img1, img2])  # ✅ Add to processed set
                pbar.update(1)  # ✅ Update progress bar

                # Save dataset metadata every `n_save_interval` images
                if image_count % n_save_interval == 0:
                    with open(output_json, "w") as f:
                        json.dump(dataset_entries, f, indent=4)
                    print(f"✅ Saved progress after {image_count} images.")

# Final Save After All Images Are Processed
with open(output_json, "w") as f:
    json.dump(dataset_entries, f, indent=4)

print(f"✅ Final dataset processing complete. Saved metadata in {output_json}.")
