import os
import numpy as np
import tifffile as tiff
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Your paths
DATA_ROOT       = "./dataset_split"               
CHECKPOINT_PATH = "./output/best_unetpp.pth"
OUT_DIR         = "./output"                 

# Configuration 
NUM_CLASSES   = 6                         
IN_CHANNELS   = 4                         
IMG_SIZE      = 256                       
BATCH_SIZE    = 16 
NUM_WORKERS   = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER_NAME  = "resnet34"               
EXCLUDE_BG_IN_MACRO = True                

CLASS_NAMES = [
    "background",  # 0
    "building",    # 1
    "road",        # 2
    "bare_land",   # 3
    "forest",      # 4
    "water"        # 5
]

assert len(CLASS_NAMES) == NUM_CLASSES, "Number of CLASS_NAMES should be equal to NUM_CLASSES."

IMG_PREFIX   = "tile_"
MASK_PREFIX  = "mask_"
EXTS_IMG     = (".tif", ".tiff", ".png")
EXTS_MASK    = (".tif", ".tiff", ".png")

os.makedirs(OUT_DIR, exist_ok=True)

# Dataset and augmentations 
def list_pairs(split_dir: str):
    img_dir  = Path(split_dir) / "img"
    mask_dir = Path(split_dir) / "mask"
    mask_map = {}
    for p in mask_dir.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS_MASK and p.name.startswith(MASK_PREFIX):
            _id = p.stem[len(MASK_PREFIX):]
            mask_map[_id] = p
    pairs = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS_IMG and p.name.startswith(IMG_PREFIX):
            _id = p.stem[len(IMG_PREFIX):]
            if _id in mask_map:
                pairs.append((str(p), str(mask_map[_id])))
    return sorted(pairs)

def load_image_4ch(path: str) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim == 2:
        raise RuntimeError(f"The image has only a single channel.: {path}")
    if arr.shape[0] in (3,4,5) and arr.ndim == 3 and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1,2,0))  # (C,H,W)->(H,W,C)
    if arr.shape[2] < 4:
        raise RuntimeError(f"The number of image channels is less than 4: {path}, shape={arr.shape}")
    if arr.shape[2] > 4:
        arr = arr[:, :, :4]
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = np.clip(arr / (arr.max() + 1e-6), 0.0, 1.0)
    return arr

def load_mask_class(path: str) -> np.ndarray:
    m = tiff.imread(path)
    if m.ndim == 3:
        if m.shape[2] == 1:
            m = m[:, :, 0]
        else:
            raise RuntimeError(f"The mask should not be multi-channel: {path}, shape={m.shape}")
    if np.issubdtype(m.dtype, np.floating):
        m = np.rint(m).astype(np.int64)
    else:
        m = m.astype(np.int64)
    return m

def get_eval_augs(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=1),
        ToTensorV2(transpose_mask=True)
    ])

class RSDataset(Dataset):
    def __init__(self, pairs, transforms=None):
        self.items = pairs
        self.tf = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        image = load_image_4ch(img_path)   # (H,W,4) float32
        mask  = load_mask_class(mask_path) # (H,W) int64
        if self.tf is not None:
            out = self.tf(image=image, mask=mask)
            image = out["image"]  # (C,H,W) tensor
            mask  = out["mask"]   # (H,W) tensor
        return image, mask.long(), img_path

# Model 
def build_model():
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,      
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_use_batchnorm=True,
    )
    return model

# Metric computation 
def confusion_matrix_numpy(y_true, y_pred, num_classes):
    """
    y_true, y_pred: (N,) int64, values in 0..num_classes-1
    Return (num_classes, num_classes) matrix: rows = ground truth, cols = prediction
    """
    cm = np.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes * num_classes
    ).reshape((num_classes, num_classes))
    return cm

def metrics_from_confusion(cm):
    """
    cm: (C,C) with rows = ground truth, cols = prediction
    Return per-class and overall metric dictionaries
    """
    C = cm.shape[0]
    TP = np.diag(cm).astype(np.float64)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        f1        = 2 * precision * recall / (precision + recall)
        iou       = TP / (TP + FP + FN)

    # Handle NaN 
    precision = np.nan_to_num(precision, nan=0.0)
    recall    = np.nan_to_num(recall,    nan=0.0)
    f1        = np.nan_to_num(f1,        nan=0.0)
    iou       = np.nan_to_num(iou,       nan=0.0)

    support   = cm.sum(axis=1)  # Number of true pixels per class
    total     = cm.sum()
    oa        = TP.sum() / total if total > 0 else 0.0

    # mIoU / mF1 (with/without background)
    if EXCLUDE_BG_IN_MACRO and C > 1:
        valid_idx = np.arange(1, C)
    else:
        valid_idx = np.arange(0, C)
    miou = iou[valid_idx].mean() if len(valid_idx) > 0 else 0.0
    mf1  = f1[valid_idx].mean() if len(valid_idx) > 0 else 0.0

    # Frequency Weighted IoU 
    freq = support / total if total > 0 else np.zeros_like(support, dtype=np.float64)
    fwiou = (freq * iou).sum()

    # Cohen's kappa
    p0 = oa
    pe = ((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (total * total) if total > 0 else 0.0
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    per_class = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "support": support.astype(np.int64)
    }
    overall = {
        "overall_accuracy": oa,
        "mIoU": float(miou),
        "mF1": float(mf1),
        "FWIoU": float(fwiou),
        "kappa": float(kappa),
        "total_pixels": int(total),
        "exclude_background_in_macro": bool(EXCLUDE_BG_IN_MACRO)
    }
    return per_class, overall

# Plotting 
def plot_metrics_bar(per_class, class_names, save_path):
    """
    Draw per-class Precision/Recall/F1/IoU bar chart
    Note: do not set specific colors/styles; use matplotlib defaults
    """
    x = np.arange(len(class_names))
    width = 0.2

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    ax.bar(x - 1.5*width, per_class["precision"], width, label="Precision")
    ax.bar(x - 0.5*width, per_class["recall"],    width, label="Recall")
    ax.bar(x + 0.5*width, per_class["f1"],        width, label="F1")
    ax.bar(x + 1.5*width, per_class["iou"],       width, label="IoU")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1 / IoU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    """
    Draw confusion matrix (rows = ground truth, cols = prediction)
    When normalize=True, row-normalize the matrix
    """
    data = cm.astype(np.float64)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        data = data / row_sums

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    im = ax.imshow(data)  # Use default colormap

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))

    # Put numbers in cells 
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                text = f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# Evaluation pipeline 
@torch.no_grad()
def evaluate():
    test_pairs = list_pairs(os.path.join(DATA_ROOT, "test"))
    if not test_pairs:
        raise RuntimeError(
    "Test set samples not found. Please check the "
    "img/ and mask/ directories under DATA_ROOT/test.")

    ds = RSDataset(test_pairs, transforms=get_eval_augs(IMG_SIZE))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

    # Build model and load weights
    model = build_model().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # Accumulate confusion matrix
    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, masks, _ in dl:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)  # (N,H,W)
        logits = model(images)                        # (N,C,H,W)
        preds  = torch.argmax(logits, dim=1)          # (N,H,W)

        # Flatten to vectors
        y_true = masks.view(-1).cpu().numpy()
        y_pred = preds.view(-1).cpu().numpy()

        # Clamp value range (clip unexpected values)
        y_true = np.clip(y_true, 0, NUM_CLASSES - 1)
        y_pred = np.clip(y_pred, 0, NUM_CLASSES - 1)

        cm = confusion_matrix_numpy(y_true, y_pred, NUM_CLASSES)
        cm_total += cm

    # Compute metrics
    per_class, overall = metrics_from_confusion(cm_total)

    # Save CSV: per-class
    df_class = pd.DataFrame({
        "class_id": np.arange(NUM_CLASSES, dtype=int),
        "class_name": CLASS_NAMES,
        "precision": per_class["precision"],
        "recall": per_class["recall"],
        "f1": per_class["f1"],
        "iou": per_class["iou"],
        "support": per_class["support"]
    })
    csv_class_path = os.path.join(OUT_DIR, "metrics_per_class.csv")
    df_class.to_csv(csv_class_path, index=False, encoding="utf-8-sig")

    # Save CSV: overall
    df_overall = pd.DataFrame([overall])
    csv_overall_path = os.path.join(OUT_DIR, "metrics_overall.csv")
    df_overall.to_csv(csv_overall_path, index=False, encoding="utf-8-sig")

    # Plotting
    plot_metrics_bar(per_class, CLASS_NAMES, os.path.join(OUT_DIR, "metrics_bar.png"))
    plot_confusion_matrix(cm_total, CLASS_NAMES, os.path.join(OUT_DIR, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(cm_total, CLASS_NAMES, os.path.join(OUT_DIR, "confusion_matrix_norm.png"), normalize=True)

    # Print key info to console
    print("\n===== Overall =====")
    for k, v in overall.items():
        print(f"{k}: {v}")
    print("\n===== Per-class =====")
    print(df_class)

    print(f"\n[Done] CSV: {csv_class_path}")
    print(f"[Done] CSV: {csv_overall_path}")
    print(f"[Done] Figures saved in: {OUT_DIR}")

if __name__ == "__main__":
    evaluate()
