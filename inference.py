import os
from typing import Tuple, List
from scipy import ndimage as ndi
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape as shp_shape
import geopandas as gpd
from tqdm import tqdm

import segmentation_models_pytorch as smp

# Your paths
IMG_PATH         = "./output/test_img/area_test1.tif" 
CHECKPOINT_PATH  = "./output/best_unetpp.pth"
OUT_DIR          = "./output/inference_out"

# Configuration 
NUM_CLASSES  = 6             
IN_CHANNELS  = 4             
ENCODER_NAME = "resnet34"    
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Sliding window params
TILE  = 256                  
STRIDE = 128                 

# TTA switch 
USE_TTA = True

# Class names 
CLASS_NAMES = [
    "background",  # 0
    "building",    # 1
    "road",        # 2
    "bare_land",   # 3
    "forest",      # 4
    "water"        # 5
]

# Utility functions 
def build_model():
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_use_batchnorm=True
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model

def hann2d(h: int, w: int) -> np.ndarray:
    """Generate a 2D Hann center weighting window (0 at borders, high at center) for logits fusion."""
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    w2d = np.outer(win_h, win_w).astype(np.float32)
    # Avoid all-zero weights (extremely small sizes) by adding a tiny epsilon
    if w2d.max() == 0:
        w2d += 1e-6
    return w2d

def norm_image(arr: np.ndarray) -> np.ndarray:
    """Normalize input (H,W,C) image to [0,1] as in training; keep only the first 4 channels."""
    if arr.ndim == 2:
        raise RuntimeError("Input image must be multi-channel (H,W,C).")
    if arr.shape[2] < 4:
        raise RuntimeError(f"Insufficient channels: {arr.shape}")
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

# TTA definition: 4 types 
def tta_transforms(x: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    """
    Input x: (1,C,H,W) tensor
    Return list[(name, x_tta)]
    """
    outs = [("id", x)]
    # Horizontal flip
    outs.append(("hflip", torch.flip(x, dims=[-1])))
    # Vertical flip
    outs.append(("vflip", torch.flip(x, dims=[-2])))
    # 90-degree rotation (counterclockwise)
    outs.append(("rot90", torch.rot90(x, k=1, dims=[-2, -1])))
    return outs

def tta_inverse(name: str, y: torch.Tensor) -> torch.Tensor:
    """
    Inverse-transform logits y: (1,C,H,W) from TTA space back to original orientation
    """
    if name == "id":
        return y
    elif name == "hflip":
        return torch.flip(y, dims=[-1])
    elif name == "vflip":
        return torch.flip(y, dims=[-2])
    elif name == "rot90":
        return torch.rot90(y, k=3, dims=[-2, -1])  # rotate back
    else:
        raise ValueError(f"Unknown TTA name: {name}")

def generate_positions(full: int, tile: int, stride: int) -> List[int]:
    """Generate sliding window starts (ensure coverage up to the right/bottom edges)."""
    if full <= tile:
        return [0]
    positions = list(range(0, full - tile, stride))
    if positions[-1] + tile < full:
        positions.append(full - tile)
    return positions

def read_window(src: rasterio.io.DatasetReader, row: int, col: int, tile: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a window image and its valid pixel mask.
    Returns:
      img: (H,W,C)
      valid: (H,W) boolean, True means valid (non-NoData)
    """
    window = Window(col_off=col, row_off=row, width=tile, height=tile)
    # Read data (auto-crop at boundaries)
    img = src.read(indexes=list(range(1, min(4, src.count) + 1)), window=window, boundless=True, fill_value=0)
    # (C,H,W) -> (H,W,C)
    img = np.transpose(img, (1, 2, 0))
    img = norm_image(img)

    # Read mask (0=nodata, 255=valid), use the first band's mask
    mask = src.read_masks(1, window=window, boundless=True)
    valid = (mask > 0)

    return img, valid

def logits_accumulators(H: int, W: int, num_classes: int):
    logits_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    return logits_sum, weight_sum

def add_logits_fused(logits_sum: np.ndarray, weight_sum: np.ndarray,
                     logits_tile: np.ndarray, weight_tile: np.ndarray,
                     row: int, col: int):
    """
    Accumulate window logits and weights into the full-size buffers.
    logits_tile: (C,h,w)
    weight_tile: (h,w)
    """
    C, h, w = logits_tile.shape
    logits_sum[:, row:row+h, col:col+w] += logits_tile * weight_tile[None, :, :]
    weight_sum[row:row+h, col:col+w]    += weight_tile

def softmax_numpy(logits: np.ndarray, axis=0) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-8, None)

def reassign_small_components(pred: np.ndarray, min_size: int = 20, num_classes: int = 6) -> np.ndarray:
    """
    For each class (including background 0), reassign connected components with pixel count < min_size
    to the most frequent neighboring class (8-connected). If a tiny island is at the image border
    with no valid neighbors, keep it unchanged.
    """
    h, w = pred.shape
    out = pred.copy()
    # 8-neighborhood structuring element
    se = np.ones((3, 3), dtype=bool)

    for k in range(num_classes):
        mask_k = (out == k)
        if not mask_k.any():
            continue
        labels, num = ndi.label(mask_k, structure=se)
        if num == 0:
            continue

        # Process each connected component
        for comp_id in range(1, num + 1):
            comp_mask = (labels == comp_id)
            area = int(comp_mask.sum())
            if area >= min_size:
                continue

            # Obtain the ring (boundary neighbors): dilate then subtract itself
            ring = ndi.binary_dilation(comp_mask, structure=se) & (~comp_mask)
            if not ring.any():
                continue  # No available neighborhood (often at outer border), skip

            neigh_vals = out[ring]
            # Count majority class in the neighborhood, excluding the current class k
            neigh_vals = neigh_vals[neigh_vals != k]
            if neigh_vals.size == 0:
                continue

            # Majority (if ties, np.bincount picks the smallest index)
            new_k = int(np.bincount(neigh_vals).argmax())
            out[comp_mask] = new_k

    return out

# Main inference pipeline 
@torch.no_grad()
def run_inference():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Open image
    with rasterio.open(IMG_PATH) as src:
        H, W = src.height, src.width
        crs = src.crs
        transform = src.transform
        dtype_out = "uint8"

        # Build model
        model = build_model()

        # Center weighting window (for full tile write-back; (256,256) weights)
        center_w = hann2d(TILE, TILE)

        # Prepare accumulators for the full canvas
        logits_sum, weight_sum = logits_accumulators(H, W, NUM_CLASSES)

        # Sliding positions
        rows = generate_positions(H, TILE, STRIDE)
        cols = generate_positions(W, TILE, STRIDE)

        for r in tqdm(rows, desc="Rows"):
            for c in cols:
                # Read window data
                img_win, valid_win = read_window(src, r, c, TILE)    # (H,W,C),(H,W)
                # (H,W,C)->(1,C,H,W)
                x = torch.from_numpy(np.transpose(img_win, (2, 0, 1))).unsqueeze(0).to(DEVICE, dtype=torch.float32)

                # Weight window: center weighting × valid pixels (NoData excluded)
                w_tile = center_w * valid_win.astype(np.float32)

                # Inference + TTA
                if USE_TTA:
                    logits_acc = torch.zeros((1, NUM_CLASSES, TILE, TILE), dtype=torch.float32, device=DEVICE)
                    weight_acc = torch.zeros((1, 1, TILE, TILE), dtype=torch.float32, device=DEVICE)
                    for name, x_t in tta_transforms(x):
                        y_t = model(x_t)  # (1,C,H,W)
                        y = tta_inverse(name, y_t)  # inverse back to original orientation
                        logits_acc += y
                        weight_acc += 1.0
                    logits = (logits_acc / torch.clamp(weight_acc, min=1e-6)).squeeze(0)  # (C,H,W)
                else:
                    logits = model(x).squeeze(0)  # (C,H,W)

                # Move to CPU -> numpy
                logits_np = logits.detach().float().cpu().numpy()  # (C,H,W)

                # Accumulate fusion
                add_logits_fused(logits_sum, weight_sum, logits_np, w_tile, r, c)

        # After fusion -> probability / class
        # If some pixels have zero weight (e.g., NoData outer ring), avoid division by zero
        weight_sum_safe = np.where(weight_sum > 0, weight_sum, 1.0)
        logits_final = logits_sum / weight_sum_safe[None, :, :]

        # softmax -> probability, then argmax
        # (Direct argmax on logits_final also works; softmax is safer here)
        probs = softmax_numpy(logits_final, axis=0)   # (C,H,W)
        pred  = np.argmax(probs, axis=0).astype(np.uint8)  # (H,W)

        # Small-component cleanup: 
        # reassign connected components <20 pixels to the surrounding class (including background) 
        pred = reassign_small_components(pred, min_size=40, num_classes=NUM_CLASSES)

        # Write prediction GeoTIFF
        pred_tif = os.path.join(OUT_DIR, "pred_class.tif")
        meta = src.meta.copy()
        meta.update({
            "count": 1,
            "dtype": dtype_out,
            "compress": "deflate",
            "predictor": 2
        })
        with rasterio.open(pred_tif, "w", **meta) as dst:
            dst.write(pred, 1)
        print(f"[Save] Prediction raster -> {pred_tif}")

        # Vectorization (polygonize): generate polygons per pixel value on the whole pred
        # Here we polygonize once and then drop background=0
        polygons = []
        values = []
        for geom, val in rasterio_shapes(pred, mask=None, transform=transform, connectivity=4):
            val = int(val)
            if val == 0:
                continue  # skip background
            shp = shp_shape(geom)
            if not shp.is_empty:
                # Filter tiny fragments (optional): e.g., area < half pixel size
                if shp.area < (abs(transform.a) * abs(transform.e)) * 0.5:
                    continue
                polygons.append(shp)
                values.append(val)

        if len(polygons) == 0:
            print("[Warn] No non-background polygons were segmented, "
                  "so writing the Shapefile was skipped.")
            return

        gdf = gpd.GeoDataFrame({
            "class_id": values,
            "class_name": [CLASS_NAMES[v] if v < len(CLASS_NAMES) else f"cls_{v}" for v in values]
        }, geometry=polygons, crs=crs)

        shp_path = os.path.join(OUT_DIR, "pred_polygons.shp")
        gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        print(f"[Save] Prediction polygons -> {shp_path}")

if __name__ == "__main__":
    run_inference()
