import os
import random
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile as tiff
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Your paths
DATA_ROOT      = "./dataset_split" 
CHECKPOINT_DIR = "./output"
LOG_DIR        = "./output"

#Training hyperparameters
NUM_CLASSES   = 6             
IN_CHANNELS   = 4             
IMG_SIZE      = 256          
BATCH_SIZE    = 16
EPOCHS        = 100
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
AMP           = True          # Mixed precision
SAVE_BEST     = True

# Choose encoder 
ENCODER_NAME      = "resnet34"
ENCODER_WEIGHTS   = None      # do not load ImageNet 
DECODER_USE_BATCHNORM = True

# Random seed 
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

# Dataset utilities 
IMG_PREFIX   = "tile_"
MASK_PREFIX  = "mask_"
EXTS_IMG     = (".tif", ".tiff", ".png")
EXTS_MASK    = (".tif", ".tiff", ".png")

def list_pairs(split_dir: str):
    img_dir  = Path(split_dir) / "img"
    mask_dir = Path(split_dir) / "mask"
    # Build mask lookup
    mask_map = {}
    for p in mask_dir.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS_MASK and p.name.startswith(MASK_PREFIX):
            _id = p.stem[len(MASK_PREFIX):]
            mask_map[_id] = p
    # Match images
    pairs = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS_IMG and p.name.startswith(IMG_PREFIX):
            _id = p.stem[len(IMG_PREFIX):]
            if _id in mask_map:
                pairs.append((str(p), str(mask_map[_id])))
    return sorted(pairs)

def load_image_4ch(path: str) -> np.ndarray:
    arr = tiff.imread(path)  # Could be (H,W,C) or (C,H,W)
    if arr.ndim == 2:
        raise RuntimeError(f"Single-channel image: {path}")
    if arr.shape[0] in (3,4,5) and arr.ndim == 3 and arr.shape[0] < arr.shape[1]:
        # (C,H,W) -> (H,W,C)
        arr = np.transpose(arr, (1,2,0))
    if arr.shape[2] < 4:
        raise RuntimeError(f"Insufficient channels (<4): {path}, shape={arr.shape}")
    if arr.shape[2] > 4:
        arr = arr[:, :, :4]

    # Normalize
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
        # Allow single-channel saved as (H,W,1)
        if m.shape[2] == 1:
            m = m[:, :, 0]
        else:
            raise RuntimeError(f"Mask should not be multi-channel: {path}, shape={m.shape}")
    # If float, convert to int
    if np.issubdtype(m.dtype, np.floating):
        m = np.rint(m).astype(np.int64)
    else:
        m = m.astype(np.int64)
    return m

# Albumentations augmentations 
def get_train_augs(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Use Affine instead of ShiftScaleRotate; use cval/cval_mask instead of value/mask_value
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-15, 15),
            interpolation=cv2.INTER_LINEAR,        # image interpolation
            mask_interpolation=cv2.INTER_NEAREST,  # mask must be nearest
            fit_output=False,
            p=0.5
        ),
        ToTensorV2(transpose_mask=True)
    ])

def get_val_augs(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        ToTensorV2(transpose_mask=True)
    ])


# Dataset 
class RSDataset(Dataset):
    def __init__(self, pairs, transforms=None):
        self.items = pairs
        self.tf = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        image = load_image_4ch(img_path)     
        mask  = load_mask_class(mask_path)   

        # albumentations expects: image(H,W,C) mask(H,W)
        if self.tf is not None:
            out = self.tf(image=image, mask=mask)
            image = out["image"]  # tensor (C,H,W)
            mask  = out["mask"]   # tensor (H,W)
        # Ensure dtype
        mask = mask.long()
        return image, mask, img_path

# Model
def build_model():
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_use_batchnorm=DECODER_USE_BATCHNORM,
    )
    return model

def compute_iou(pred, target, num_classes=6, ignore_index=None):
    ious = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_c = (pred == cls)
        targ_c = (target == cls)
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / (union + 1e-7))
    if not ious:
        return 0.0
    return float(np.mean(ious))

# Train
def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for images, masks, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP):
            logits = model(images)                 
            loss = criterion(logits, masks)        

        if AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(1, n_batches)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    n_batches = 0
    miou_sum = 0.0
    n_samples = 0
    for images, masks, _ in tqdm(loader, desc="Val", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=AMP):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1)  # (N,H,W)
        miou_sum += compute_iou(preds.cpu(), masks.cpu(), num_classes=NUM_CLASSES)
        n_samples += 1
    return running_loss / max(1, n_batches), miou_sum / max(1, n_samples)

# Main
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_pairs = list_pairs(os.path.join(DATA_ROOT, "train"))
    val_pairs   = list_pairs(os.path.join(DATA_ROOT, "val"))
    test_pairs  = list_pairs(os.path.join(DATA_ROOT, "test"))

    print(f"[Info] train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    train_ds = RSDataset(train_pairs, transforms=get_train_augs(IMG_SIZE))
    val_ds   = RSDataset(val_pairs,   transforms=get_val_augs(IMG_SIZE))
    test_ds  = RSDataset(test_pairs,  transforms=get_val_augs(IMG_SIZE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_miou = -1.0
    best_path = os.path.join(CHECKPOINT_DIR, "best_unetpp.pth")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        val_loss, val_miou = eval_one_epoch(model, val_loader, criterion)
        scheduler.step()

        print(
            f"[Epoch {epoch}] "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mIoU={val_miou:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Log and save
        with open(os.path.join(LOG_DIR, "log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{tr_loss:.6f}\t{val_loss:.6f}\t{val_miou:.6f}\n")

        if SAVE_BEST and val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_mIoU": best_miou,
                "config": {
                    "in_channels": IN_CHANNELS,
                    "num_classes": NUM_CLASSES,
                    "encoder": ENCODER_NAME
                }
            }, best_path)
            print(f"[Save] New best mIoU={best_miou:.4f} -> {best_path}")

    # Save last epoch
    final_path = os.path.join(CHECKPOINT_DIR, "last_unetpp.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[Save] Final model -> {final_path}")

    # Test set evaluation 
    print("\n[Eval] Testing best model on test set ...")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
    test_loss, test_miou = eval_one_epoch(model, test_loader, criterion)
    print(f"[Test] loss={test_loss:.4f} | mIoU={test_miou:.4f}")

if __name__ == "__main__":
    main()
