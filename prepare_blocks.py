import numpy as np
import torch
import os
from collections import Counter

# Path to the original data
path = '/Train_Fold_1.npy'
# path = '/Train_Fold_2.npy'
data = np.load(path)


save_dir = 'data/dfc2018_fold1/train'
# save_dir = 'data/dfc2018_fold1/val'
os.makedirs(save_dir, exist_ok=True)

print("Original data shape:", data.shape)
print("XYZ example:", data[0, :3])
print("RGB example:", data[0, 3:6])
print("HSI example:", data[0, 6:-1])
print("Label example:", data[0, -1])

# Separate components
xyz = data[:, :3]               # (N, 3)
spectral = data[:, 3:-1]        # (N, 12): RGB + Intensity + 8 HSI
labels = data[:, -1].astype(int)  # (N,)
labels_remap = labels.copy()
labels_remap[labels_remap == 0] = -1      # unlabeled → ignore
labels_remap[labels_remap > 0] -= 1       # valid classes 1–18 → 0–17

# Preprocess spectral data: scale 16-bit → 0–1
spectral = spectral.astype(np.float32) / 255.0  # keep float precision

# Combine XYZ + spectral
scene_points = np.hstack((xyz, spectral))
print("Scene points shape:", scene_points.shape)  # (N, 15)

# Block parameters
block_size = 75
stride = 75
padding = 0
min_points = 1000

coord_min = np.amin(xyz, axis=0)
coord_max = np.amax(xyz, axis=0)
grid_x = int(np.ceil((coord_max[0] - coord_min[0] - block_size) / stride) + 1)
grid_y = int(np.ceil((coord_max[1] - coord_min[1] - block_size) / stride) + 1)

block_id = 0
for idx_y in range(grid_y):
    for idx_x in range(grid_x):
        s_x = coord_min[0] + idx_x * stride
        e_x = min(s_x + block_size, coord_max[0])
        s_x = e_x - block_size
        s_y = coord_min[1] + idx_y * stride
        e_y = min(s_y + block_size, coord_max[1])
        s_y = e_y - block_size

        # Get indices inside block
        point_idxs = np.where(
            (xyz[:, 0] >= s_x - padding) & (xyz[:, 0] <= e_x + padding) &
            (xyz[:, 1] >= s_y - padding) & (xyz[:, 1] <= e_y + padding)
        )[0]

        if point_idxs.size < min_points:
            continue

        # Shuffle points within block
        np.random.shuffle(point_idxs)

        data_block = scene_points[point_idxs, :]
        label_block = labels_remap[point_idxs]

        # Keep original XYZ for mapping predictions back
        data_pth = {
            'coord': data_block[:, :3],      # original XYZ (not normalized)
            'color': data_block[:, 3:],      # spectral 12D 
            'semantic_gt': label_block[:, None],
            'original_coord': xyz[point_idxs, :],  # optional, identical to coord here
        }

        torch.save(data_pth, os.path.join(save_dir, f'Block_{block_id}.pth'))

        # Print block info
        class_counts = Counter(label_block)
        print(f"Block {block_id}: {point_idxs.size} points, Classes: {dict(class_counts)}")

        block_id += 1

print(f"Saved {block_id} blocks.")
