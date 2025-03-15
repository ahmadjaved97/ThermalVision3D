import os
import glob
import numpy as np
import torch
import json
import cv2
from tqdm import tqdm
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

# 📂 Define dataset root
FREIBURG_PATH = "/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg/train"
SAVE_PATH = "/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg/Pseudo_GT"

# 🔹 Load MASt3r Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

# 🔹 Iterate over all sequences
for seq_name in sorted(os.listdir(FREIBURG_PATH)):  
    seq_path = os.path.join(FREIBURG_PATH, seq_name)

    if not os.path.isdir(seq_path):  # Skip non-folder files
        continue

    # Iterate over all sub-sequences (00, 01, ...)
    for sub_seq in sorted(os.listdir(seq_path)):
        sub_seq_path = os.path.join(seq_path, sub_seq, "fl_rgb")  # Path to RGB images

        if not os.path.exists(sub_seq_path):  # Skip if no RGB folder
            continue
        
        print(f"Processing: {seq_name}/{sub_seq}")

        # 📂 Define save paths in Co3Dv2 format
        SEQUENCE_SAVE_PATH = os.path.join(SAVE_PATH, f"{seq_name}_{sub_seq}")
        IMAGE_PATH = os.path.join(SEQUENCE_SAVE_PATH, "images")
        DEPTH_PATH = os.path.join(SEQUENCE_SAVE_PATH, "depths")
        POSE_PATH = os.path.join(SEQUENCE_SAVE_PATH, "poses")
        os.makedirs(IMAGE_PATH, exist_ok=True)
        os.makedirs(DEPTH_PATH, exist_ok=True)
        os.makedirs(POSE_PATH, exist_ok=True)

        # 🔹 Load images
        rgb_images = sorted(glob.glob(os.path.join(sub_seq_path, "*.png")))

        if len(rgb_images) < 2:
            print(f"Skipping {seq_name}/{sub_seq}, not enough images")
            continue

        # 🔹 Process consecutive image pairs
        for i in tqdm(range(len(rgb_images) - 1)):
            img1_path, img2_path = rgb_images[i], rgb_images[i + 1]
            frame1 = os.path.basename(img1_path).split('.')[0]
            frame2 = os.path.basename(img2_path).split('.')[0]

            # Load images and preprocess for MASt3r
            images = load_images([img1_path, img2_path], size=512)

            # Pass through MASt3r
            output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

            # Extract predictions
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']

            # Extract depth maps
            depth_map = pred1['depth'].squeeze(0).cpu().numpy()
            np.save(os.path.join(DEPTH_PATH, f"{frame1}.npy"), depth_map)
            print(f"Saved Depth: {frame1}.npy")

            # Extract pose transformation
            desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device, dist='dot', block_size=2**13)

            # Compute valid matches
            H0, W0 = view1['true_shape'][0]
            valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)
            H1, W1 = view2['true_shape'][0]
            valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)
            valid_matches = valid_matches_im0 & valid_matches_im1
            matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

            # Generate pose transformation matrix (Identity for now, replace with actual computation)
            pose_matrix = np.eye(4)

            # Save Pose Matrix
            pose_data = {
                "rotation": pose_matrix[:3, :3].tolist(),
                "translation": pose_matrix[:3, 3].tolist()
            }
            with open(os.path.join(POSE_PATH, f"{frame1}_to_{frame2}.json"), "w") as f:
                json.dump(pose_data, f, indent=4)
            print(f"Saved Pose: {frame1}_to_{frame2}.json")

        # 🔹 Save Camera Intrinsics (Assumed from metadata.json if available)
        intrinsics_file = os.path.join(seq_path, sub_seq, "metadata.json")
        if os.path.exists(intrinsics_file):
            with open(intrinsics_file, "r") as f:
                intrinsics = json.load(f)
        else:
            intrinsics = {
                "fx": 718.856, "fy": 718.856,
                "cx": 607.1928, "cy": 185.2157,
                "skew": 0.0
            }
        with open(os.path.join(SEQUENCE_SAVE_PATH, "intrinsics.json"), "w") as f:
            json.dump(intrinsics, f, indent=4)
        print("Saved Camera Intrinsics")

print("✅ All sequences processed!")

