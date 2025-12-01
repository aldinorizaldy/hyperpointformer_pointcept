# HyperPointFormer (Version 2)

This is a second version of **HyperPointFormer**, built on top of the **PointCept** codebase.  
The main advantage of this version is its ability to efficiently process **large point clouds** while maintaining strong segmentation accuracy.

---

## Performance Comparison

### Validation Results

| Model | mIoU | mAcc | allAcc |
|-------|------|------|--------|
| Point Transformer (Baseline) | 0.4072 | 0.5385 | 0.7594 |
| HyperPointFormer | 0.5212 | 0.6378 | 0.7891 |

---

### Per-Class IoU Comparison

| Class | Point Transformer IoU | HyperPointFormer IoU |
|-------|---------------------|--------------------|
| Healthy grass | 0.1849 | 0.6420 |
| Stressed grass | 0.3968 | 0.7286 |
| Evergreen trees | 0.7225 | 0.9200 |
| Deciduous trees | 0.4979 | 0.8195 |
| Bare earth | 0.0559 | 0.5479 |
| Water | 0.0000 | 0.0000 |
| Residential buildings | 0.4369 | 0.3435 |
| Non-residential buildings | 0.8424 | 0.8263 |
| Roads | 0.5142 | 0.4662 |
| Sidewalks | 0.5905 | 0.5806 |
| Crosswalks | 0.0000 | 0.0000 |
| Major thoroughfares | 0.4693 | 0.3929 |
| Highways | 0.4372 | 0.3815 |
| Railways | 0.2141 | 0.2903 |
| Paved parking lots | 0.5771 | 0.5609 |
| Unpaved parking lots | 0.0000 | 0.2688 |
| Cars | 0.6269 | 0.7751 |
| Trains | 0.7624 | 0.8370 |

---

This version leverages **PointCept** to handle large-scale point clouds more efficiently, resulting in **improved segmentation metrics** compared to the baseline Point Transformer.

---

## Data Preparation (IEEE GRSS DFC 2018)

To prepare the IEEE DFC 2018 dataset for training HyperPointFormer, follow the steps below.

### 1. Download the Raw Data
The preprocessed DFC2018 dataset can be downloaded from [here](link)


### 2. Run Data Preparation
Use the provided script to convert the npy point cloud data into block files:

```bash
python prepare_blocks.py 
```

### 3. Training
Then follow Pointcept training pipeline

```bash
train.sh -g 4 -d dfc2018 -c semseg-hyperpointformer -n dfc2018_semseg-hpf-0-base
```