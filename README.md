# U-Net++ Semantic-Segmentation on Multispectral Satellite Imagery
This project uses open-source **NAIP** satellite imagery to build a small land-cover classification dataset and train a **U-Net++** model for segmentation. The results were evaluated on multiple land-cover classes, and **single-image inference** is supported for visualization and testing.

## Dataset 
The data source is aerial orthophotography from the U.S. Department of Agriculture’s National Agriculture Imagery Program (NAIP), with a spatial resolution of **60 cm** and **four spectral bands (RGB + NIR)**. A total of **215 images** were produced as training data, with the dataset split in a **6:2:2 ratio** for training, validation, and testing. Each image has a size of **256 × 256 pixels**, and there are **six land-cover classes(0:background; 1:buiding; 2:road; 3:bare land; 4:forest; 5:water)** in total.

Example of Training Image and Mask
<img width="1084" height="443" alt="eg" src="https://github.com/user-attachments/assets/65220262-e619-408f-8557-f4f82bf66e8a" />
Dataset Structure
```bash
└── dataset_split
    ├── train
    |   ├── img
    │   ├── mask
    └── val
    |   ├── img
    │   ├── mask
    └── test
        ├── img
        ├── mask
```
## Installation
The training was conducted on a **Windows 10** equipped with **CUDA 11.3**.
```bash
conda create --name smp python=3.9 -y 
conda activate mmrotate
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install segmentation-models-pytorch
```

## Train
The model U-Net++ from **SMP (segmentation_models.pytorch)** 
https://github.com/qubvel-org/segmentation_models.pytorch was used for training.

```bash
python train.py
```

The training curves
<img width="2000" height="1200" alt="training_curves" src="https://github.com/user-attachments/assets/9323eaee-b44c-440f-9e4c-314b80b5c42d" />

## Evaluate
The evaluation was performed using **per-class and overall metrics** on the test set.

```bash
python evaluate.py
```
### Overall Metrics
|overall_accuracy|mIoU|mF1|kappa|
|:----------------:|:----:|:----:|:----:|
|0.90|0.78|0.88|0.84|

### Per-class Metrics
|class_id|class_name|precision|recall|f1|iou|
|:----:|:--------:|:----:|:----:|:----:|:----:|
|0|background|0.93|0.88|0.90|0.82|
|1|building|0.81|0.78|0.79|0.66|
|2|road|0.85|0.82|0.83|0.71|
|3|bare land|0.84|0.96|0.90|0.81|
|4|forest|0.89|0.89|0.89|0.80|
|5|water|0.95|0.98|0.97|0.93|

### Confusion Matrix
<img width="1600" height="1400" alt="confusion_matrix_norm" src="https://github.com/user-attachments/assets/bde40f18-3899-4dc6-9d7e-2cd956cc944f" />

## Single-image inference
Single-image inference uses 256×256 sliding windows with **50% overlap**, performs **logit-level** fusion with a **Hann center weighting**, and applies a **global softmax→argmax** for seamless outputs. **TTA**(horizontal/vertical flips and 90° rotation) is included by inverse-transforming and accumulating logits. Finally, **a small-component reassignment** (connected regions <20 px merged to the neighborhood majority) removes holes and speckle.

```bash
python inference.py
```

**Inference was performed on a test image with a size of 1024 × 1024 pixels.**


<img width="1294" height="466" alt="all" src="https://github.com/user-attachments/assets/1b0d2ee7-9963-4951-9096-960a8ff1d160" />

## 📜 Data Usage and Attribution

This dataset was created using open-source satellite imagery from the **U.S. Department of Agriculture National Agriculture Imagery Program (NAIP)**.

If you use this dataset in your research or project, please cite or acknowledge:

> Source imagery: USDA NAIP  
> Processed and labeled dataset: yinx111 (2025), https://github.com/yinx111/U-Net-Semantic-Segmentation-on-Multispectral-RGB-NIR-Imagery

**And the dataset will be continuously expanded with additional land-cover categories and samples in future updates.**


