"""
 AI Imaging Coding Case Study
Complete end-to-end biomedical image-analysis pipeline.


IMPORTANT:
- Educational use only. This script does not produce clinical diagnoses.
- Ollama must be running locally for LLM steps.
- Recommended models:
    ollama pull llama3.2-vision
    ollama pull llama3.2
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

from skimage import filters, measure, morphology
from skimage.io import imread
from skimage.transform import resize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



# Configuration


SEED = 42
IMAGE_SIZE = 256

DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "Nickolay-K/Assingnment-3-dataset/main/nuclei_dataset.zip"
)

VISION_MODEL = "llama3.2-vision"
TEXT_MODEL = "llama3.2"

DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-3

# Run all three to produce the requested model/loss comparison table.
LOSSES_TO_RUN = ("bce", "dice", "bce_dice")



# Reproducibility and utilities


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_json_loads(text: str) -> Dict:
    """Extract a JSON object even if a model adds a small amount of extra text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def device_name() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Dataset download and discovery


def download_dataset(zip_path: Path) -> None:
    if zip_path.exists():
        print(f"[Data] Using existing archive: {zip_path}")
        return

    print(f"[Data] Downloading dataset to {zip_path} ...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    print("[Data] Download complete.")


def extract_dataset(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"[Data] Dataset already extracted: {extract_dir}")
        return

    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    print(f"[Data] Extracted dataset to {extract_dir}")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_files_under(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def looks_like_mask_path(path: Path) -> bool:
    s = str(path).lower()
    keys = ("mask", "masks", "label", "labels", "segmentation", "ground_truth", "groundtruth")
    return any(k in s for k in keys)


def looks_like_image_path(path: Path) -> bool:
    s = str(path).lower()
    keys = ("image", "images", "img", "imgs")
    return any(k in s for k in keys) and not looks_like_mask_path(path)


def normalize_stem(stem: str) -> str:
    s = stem.lower()
    suffixes = [
        "_mask", "-mask", " mask",
        "_masks", "-masks",
        "_label", "-label",
        "_labels", "-labels",
        "_seg", "-seg",
        "_gt", "-gt",
    ]
    for suf in suffixes:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


@dataclass
class Sample:
    image: Path
    mask: Optional[Path]


def pair_images_and_masks(root: Path) -> List[Sample]:
    all_files = image_files_under(root)
    if not all_files:
        raise FileNotFoundError(f"No image files found under {root}")

    mask_files = [p for p in all_files if looks_like_mask_path(p)]
    image_files = [p for p in all_files if p not in mask_files]

    # If folder naming is unusual, use filename heuristics.
    if not mask_files:
        candidate_masks = []
        candidate_images = []
        for p in all_files:
            stem = p.stem.lower()
            if any(k in stem for k in ("mask", "label", "_gt", "-gt")):
                candidate_masks.append(p)
            else:
                candidate_images.append(p)
        if candidate_masks:
            mask_files = candidate_masks
            image_files = candidate_images

    mask_map: Dict[str, List[Path]] = {}
    for m in mask_files:
        mask_map.setdefault(normalize_stem(m.stem), []).append(m)

    samples = []
    for img in image_files:
        key = normalize_stem(img.stem)
        matches = mask_map.get(key, [])

        # Also try matching same parent hierarchy / close filename if no exact match.
        if not matches:
            matches = [
                m for m in mask_files
                if normalize_stem(m.stem) == key
                or key in normalize_stem(m.stem)
                or normalize_stem(m.stem) in key
            ]

        mask = matches[0] if matches else None
        samples.append(Sample(img, mask))

    # Remove likely mask files accidentally included as images.
    samples = [s for s in samples if not looks_like_mask_path(s.image)]
    return samples


def detect_named_split(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    if any(p in {"train", "training"} for p in parts):
        return "train"
    if any(p in {"val", "valid", "validation"} for p in parts):
        return "val"
    if any(p in {"test", "testing"} for p in parts):
        return "test"
    return None


def split_samples(
    samples: List[Sample],
    seed: int = SEED,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Prefer dataset-provided train/val/test folder names.
    Otherwise create deterministic 70/15/15 splits.
    """
    named = {"train": [], "val": [], "test": [], None: []}
    for s in samples:
        split = detect_named_split(s.image)
        named[split].append(s)

    if named["train"] and (named["val"] or named["test"]):
        train = named["train"]
        val = named["val"]
        test = named["test"]

        leftovers = named[None]
        # If one split is missing, derive it from train/leftovers.
        rng = random.Random(seed)
        if leftovers:
            rng.shuffle(leftovers)
            test.extend(leftovers)

        if not val:
            rng.shuffle(train)
            n_val = max(1, int(round(0.15 * len(train))))
            val = train[:n_val]
            train = train[n_val:]

        if not test:
            rng.shuffle(train)
            n_test = max(1, int(round(0.15 * len(train))))
            test = train[:n_test]
            train = train[n_test:]

        return train, val, test

    rng = random.Random(seed)
    samples = samples[:]
    rng.shuffle(samples)

    n = len(samples)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    n_train = max(1, n - n_val - n_test)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    return train, val, test



# Image preprocessing


def read_grayscale(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    arr = imread(path)

    if arr.ndim == 3:
        # RGB/RGBA -> luminance-like mean, robust to arbitrary channel counts.
        arr = arr[..., :3].astype(np.float32)
        arr = (
            0.2126 * arr[..., 0]
            + 0.7152 * arr[..., 1]
            + 0.0722 * arr[..., 2]
        )

    arr = arr.astype(np.float32)

    # Normalize before resize.
    amin, amax = float(np.min(arr)), float(np.max(arr))
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    arr = resize(
        arr,
        (size, size),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    return np.clip(arr, 0.0, 1.0)


def read_binary_mask(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    arr = imread(path)

    if arr.ndim == 3:
        arr = arr[..., 0]

    arr = arr.astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()

    arr = resize(
        arr,
        (size, size),
        preserve_range=True,
        anti_aliasing=False,
        order=0,
    )
    return (arr > 0.5).astype(np.float32)


def save_processed_images(samples: Sequence[Sample], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for sample in samples:
        arr = read_grayscale(sample.image)
        out_path = out_dir / f"{sample.image.stem}.png"
        Image.fromarray((arr * 255).astype(np.uint8)).save(out_path)



#  EDA + VLM


def make_eda(samples: Sequence[Sample], out_dir: Path, n_show: int = 6) -> None:
    ensure_dir(out_dir)
    subset = list(samples[: min(n_show, len(samples))])

    # Sample images
    cols = 3
    rows = math.ceil(len(subset) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, s in zip(axes, subset):
        arr = read_grayscale(s.image)
        ax.imshow(arr, cmap="gray")
        ax.set_title(s.image.name, fontsize=8)
        ax.axis("off")

    fig.suptitle("EDA: representative grayscale 256×256 images")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_sample_images.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Intensity histogram across a sample of the dataset.
    pixel_values = []
    for s in samples[: min(50, len(samples))]:
        pixel_values.append(read_grayscale(s.image).ravel())
    vals = np.concatenate(pixel_values)

    plt.figure(figsize=(7, 4.5))
    plt.hist(vals, bins=50)
    plt.xlabel("Normalized grayscale intensity")
    plt.ylabel("Pixel count")
    plt.title("EDA: intensity histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_intensity_histogram.png", dpi=180)
    plt.close()


NAIVE_VLM_PROMPT = """
Describe this biomedical image.
""".strip()


OPTIMIZED_VLM_PROMPT = """
You are analysing a biomedical microscopy image for an educational image-analysis
assignment. Describe only visible image characteristics. Do NOT diagnose disease,
infer patient identity, or make clinical claims.

Return exactly one valid JSON object with these keys:
{
  "modality": "short modality description or uncertain",
  "tissue_type": "visible tissue/cell type or uncertain",
  "notable_features": ["brief visible feature 1", "brief visible feature 2"],
  "image_quality": "good | moderate | poor | uncertain"
}

Rules:
- Base every statement only on what is visually observable.
- If evidence is insufficient, use "uncertain".
- Do not include markdown.
- Do not include text before or after the JSON.
""".strip()


def image_to_base64(path: Path) -> str:
    arr = read_grayscale(path)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ollama_chat(
    model: str,
    prompt: str,
    image_path: Optional[Path] = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    timeout: int = 180,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature},
    }

    if image_path is not None:
        payload["messages"][0]["images"] = [image_to_base64(image_path)]

    if json_mode:
        payload["format"] = "json"

    r = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def task1_vlm(
    representative_image: Path,
    out_dir: Path,
    repeat_runs: int = 3,
) -> Dict:
    ensure_dir(out_dir)

    result = {
        "representative_image": str(representative_image),
        "naive_prompt": NAIVE_VLM_PROMPT,
        "optimized_prompt": OPTIMIZED_VLM_PROMPT,
        "naive_output": None,
        "optimized_runs": [],
        "ollama_error": None,
    }

    try:
        naive = ollama_chat(
            VISION_MODEL,
            NAIVE_VLM_PROMPT,
            image_path=representative_image,
            temperature=0.3,
            json_mode=False,
        )
        result["naive_output"] = naive

        # Slightly nonzero temperature demonstrates run-to-run variability.
        for i in range(repeat_runs):
            text = ollama_chat(
                VISION_MODEL,
                OPTIMIZED_VLM_PROMPT,
                image_path=representative_image,
                temperature=0.35,
                json_mode=True,
            )
            try:
                parsed = safe_json_loads(text)
            except Exception:
                parsed = {"raw_output": text, "valid_json": False}
            result["optimized_runs"].append(parsed)

    except Exception as e:
        result["ollama_error"] = (
            f"{type(e).__name__}: {e}. "
            "Make sure Ollama is running and llama3.2-vision is installed."
        )

    save_json(result, out_dir / "task1_vlm_results.json")
    return result



#  Classical segmentation , region features and LLM


REGION_COLUMNS = [
    "label",
    "area",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
]


def otsu_segment(image: np.ndarray) -> np.ndarray:
    threshold = filters.threshold_otsu(image)
    binary = image > threshold

    # If foreground occupies most of the image, invert. This makes the
    # procedure more robust across bright-on-dark and dark-on-bright modalities.
    if binary.mean() > 0.65:
        binary = ~binary

    min_size = max(16, int(0.0005 * image.size))
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=min_size)
    binary = morphology.binary_opening(binary, morphology.disk(1))
    binary = morphology.binary_closing(binary, morphology.disk(2))
    return binary.astype(bool)


def region_feature_table(
    image: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    labels = measure.label(mask)

    props = measure.regionprops_table(
        labels,
        intensity_image=image,
        properties=(
            "label",
            "area",
            "eccentricity",
            "solidity",
            "mean_intensity",
            "perimeter",
            "major_axis_length",
            "minor_axis_length",
        ),
    )
    df = pd.DataFrame(props)
    if df.empty:
        return pd.DataFrame(columns=REGION_COLUMNS)
    return df


def summarize_features_numbers_only(df: pd.DataFrame, mask: np.ndarray) -> str:
    if df.empty:
        return (
            "n_objects=0; foreground_fraction="
            f"{float(mask.mean()):.4f}; no object-level measurements available."
        )

    return (
        f"n_objects={len(df)}; "
        f"foreground_fraction={float(mask.mean()):.4f}; "
        f"area_mean={df['area'].mean():.2f}; "
        f"area_median={df['area'].median():.2f}; "
        f"area_std={df['area'].std(ddof=0):.2f}; "
        f"eccentricity_mean={df['eccentricity'].mean():.3f}; "
        f"solidity_mean={df['solidity'].mean():.3f}; "
        f"mean_intensity_mean={df['mean_intensity'].mean():.3f}; "
        f"perimeter_mean={df['perimeter'].mean():.2f}; "
        f"major_axis_mean={df['major_axis_length'].mean():.2f}; "
        f"minor_axis_mean={df['minor_axis_length'].mean():.2f}."
    )


NUMBERS_ONLY_PROMPT_TEMPLATE = """
You are given numerical measurements extracted from a biomedical image.
You do NOT have access to the image. Do not diagnose disease or invent visual
details that are not supported by the measurements.

Measurements:
{summary}

Return exactly one valid JSON object:
{{
  "n_objects": <integer>,
  "density_class": "low | moderate | high | uncertain",
  "shape_regularity": "regular | mixed | irregular | uncertain",
  "quality_flag": "good | review | uncertain",
  "description": "one short paragraph based only on the supplied numbers"
}}

Use "uncertain" where the measurements do not justify a confident label.
Do not include markdown or any text outside the JSON.
""".strip()


def task2_classical(
    image_path: Path,
    out_dir: Path,
) -> Dict:
    ensure_dir(out_dir)

    image = read_grayscale(image_path)
    mask = otsu_segment(image)
    df = region_feature_table(image, mask)
    summary = summarize_features_numbers_only(df, mask)

    stem = image_path.stem
    df.to_csv(out_dir / f"{stem}_regionprops.csv", index=False)

    # Save visual check.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Otsu + morphology")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_classical_segmentation.png", dpi=180)
    plt.close(fig)

    prompt = NUMBERS_ONLY_PROMPT_TEMPLATE.format(summary=summary)
    output = None
    error = None

    try:
        text = ollama_chat(
            TEXT_MODEL,
            prompt,
            image_path=None,
            temperature=0.2,
            json_mode=True,
        )
        output = safe_json_loads(text)
    except Exception as e:
        error = (
            f"{type(e).__name__}: {e}. "
            "Make sure Ollama is running and the text model is installed."
        )

    result = {
        "image": str(image_path),
        "feature_summary": summary,
        "prompt": prompt,
        "llm_output": output,
        "ollama_error": error,
    }
    save_json(result, out_dir / f"{stem}_task2_numbers_first.json")
    return result



#  PyTorch U-Net


class NucleiDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = [s for s in samples if s.mask is not None]
        if not self.samples:
            raise ValueError(
                "No image/mask pairs were found. "
                "Check the extracted dataset structure."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = read_grayscale(sample.image)
        mask = read_binary_mask(sample.mask)

        x = torch.from_numpy(image[None, ...]).float()
        y = torch.from_numpy(mask[None, ...]).float()
        return x, y, str(sample.image), str(sample.mask)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


def dice_coefficient_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    preds = (probs >= threshold).float()
    dims = tuple(range(1, preds.ndim))
    inter = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)
    return ((2 * inter + eps) / (denom + eps)).mean()


def iou_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    preds = (probs >= threshold).float()
    dims = tuple(range(1, preds.ndim))
    inter = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - inter
    return ((inter + eps) / (union + eps)).mean()


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss(logits, targets)

    if loss_name == "bce":
        return bce
    if loss_name == "dice":
        return dice
    if loss_name == "bce_dice":
        return bce + dice
    raise ValueError(f"Unknown loss: {loss_name}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    dices, ious = [], []

    for x, y, *_ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)

        dices.append(float(dice_coefficient_from_probs(probs, y).item()))
        ious.append(float(iou_from_probs(probs, y).item()))

    return {
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
    }


def train_one_model(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    out_dir: Path,
    loss_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Tuple[SmallUNet, pd.DataFrame, Dict[str, float]]:
    ensure_dir(out_dir)
    device = device_name()
    print(f"[U-Net:{loss_name}] Device: {device}")

    train_ds = NucleiDataset(train_samples)
    val_ds = NucleiDataset(val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_dice = -1.0
    best_path = out_dir / f"unet_{loss_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for x, y, *_ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = segmentation_loss(logits, y, loss_name)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        metrics = evaluate_model(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": metrics["dice"],
            "val_iou": metrics["iou"],
        }
        history.append(row)

        print(
            f"[U-Net:{loss_name}] "
            f"epoch={epoch:02d}/{epochs} "
            f"loss={row['train_loss']:.4f} "
            f"dice={row['val_dice']:.4f} "
            f"iou={row['val_iou']:.4f}"
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(model.state_dict(), best_path)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / f"history_{loss_name}.csv", index=False)

    # Restore best validation checkpoint.
    model.load_state_dict(torch.load(best_path, map_location=device))
    final_metrics = evaluate_model(model, val_loader, device)

    return model, hist_df, final_metrics


def plot_training_curves(
    histories: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)

    plt.figure(figsize=(7, 4.5))
    for loss_name, df in histories.items():
        plt.plot(df["epoch"], df["train_loss"], label=loss_name)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("U-Net training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "unet_loss_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for loss_name, df in histories.items():
        plt.plot(df["epoch"], df["val_dice"], label=loss_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.title("U-Net validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "unet_dice_curves.png", dpi=180)
    plt.close()


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    image = read_grayscale(image_path)
    x = torch.from_numpy(image[None, None, ...]).float().to(device)
    probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = probs >= 0.5
    return probs, mask


@torch.no_grad()
def save_validation_panels(
    model: nn.Module,
    val_samples: Sequence[Sample],
    out_dir: Path,
    n: int = 3,
) -> None:
    ensure_dir(out_dir)
    device = device_name()
    model.eval()

    candidates = [s for s in val_samples if s.mask is not None][:n]

    for i, s in enumerate(candidates, start=1):
        image = read_grayscale(s.image)
        gt = read_binary_mask(s.mask)
        _, pred = predict_mask(model, s.image, device)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground truth")
        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("U-Net prediction")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(s.image.name)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"validation_panel_{i}_{s.image.stem}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def evaluate_otsu_against_ground_truth(
    samples: Sequence[Sample],
) -> Dict[str, float]:
    dices, ious = [], []

    for s in samples:
        if s.mask is None:
            continue

        image = read_grayscale(s.image)
        gt = read_binary_mask(s.mask).astype(bool)
        pred = otsu_segment(image)

        inter = np.logical_and(pred, gt).sum()
        denom = pred.sum() + gt.sum()
        union = np.logical_or(pred, gt).sum()

        dice = (2 * inter + 1e-7) / (denom + 1e-7)
        iou = (inter + 1e-7) / (union + 1e-7)

        dices.append(float(dice))
        ious.append(float(iou))

    return {
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
    }


def find_best_examples(
    model: nn.Module,
    val_samples: Sequence[Sample],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Find an image where U-Net improves most over Otsu and one where Otsu
    does better. This directly supports one of the required report questions.
    """
    ensure_dir(out_dir)
    device = device_name()
    model.eval()
    rows = []

    for s in val_samples:
        if s.mask is None:
            continue

        image = read_grayscale(s.image)
        gt = read_binary_mask(s.mask).astype(bool)
        otsu = otsu_segment(image)
        _, unet = predict_mask(model, s.image, device)

        def dice_np(pred, target):
            inter = np.logical_and(pred, target).sum()
            denom = pred.sum() + target.sum()
            return float((2 * inter + 1e-7) / (denom + 1e-7))

        otsu_d = dice_np(otsu, gt)
        unet_d = dice_np(unet, gt)

        rows.append({
            "image": str(s.image),
            "otsu_dice": otsu_d,
            "unet_dice": unet_d,
            "unet_minus_otsu": unet_d - otsu_d,
        })

    df = pd.DataFrame(rows).sort_values("unet_minus_otsu", ascending=False)
    df.to_csv(out_dir / "unet_vs_otsu_per_image.csv", index=False)

    if not df.empty:
        examples = {
            "unet_better": df.iloc[0].to_dict(),
            "otsu_better": df.iloc[-1].to_dict(),
        }
        save_json(examples, out_dir / "unet_vs_otsu_examples.json")

    return df



#  Hybrid pipeline


HYBRID_JSON_PROMPT = """
You are given numerical measurements extracted from a U-Net segmentation mask.
You do not see the image. Do not diagnose disease and do not invent details.

image_id={image_id}
n_objects={n_objects}
mean_area={mean_area:.2f}
foreground_fraction={foreground_fraction:.4f}
mean_eccentricity={mean_eccentricity:.3f}
mean_solidity={mean_solidity:.3f}

Return exactly one valid JSON object:
{{
  "image_id": "{image_id}",
  "n_objects": {n_objects},
  "mean_area": {mean_area:.2f},
  "density_class": "low | moderate | high | uncertain",
  "quality_flag": "good | review | uncertain"
}}

The numeric fields image_id, n_objects and mean_area must agree with the
measurements supplied above. Do not add markdown or other text.
""".strip()


NARRATIVE_PROMPT = """
Convert the structured record below into one concise paragraph for an
educational biomedical image-analysis report.

Do not diagnose disease. Do not add facts that are absent from the record.
If a field is uncertain, preserve that uncertainty.

Structured record:
{record}
""".strip()


def deterministic_hybrid_fallback(
    image_id: str,
    df: pd.DataFrame,
    mask: np.ndarray,
) -> Dict:
    n_objects = int(len(df))
    mean_area = float(df["area"].mean()) if not df.empty else 0.0
    frac = float(mask.mean())

    if frac < 0.08:
        density = "low"
    elif frac < 0.25:
        density = "moderate"
    else:
        density = "high"

    quality = "review" if n_objects == 0 else "good"

    return {
        "image_id": image_id,
        "n_objects": n_objects,
        "mean_area": round(mean_area, 2),
        "density_class": density,
        "quality_flag": quality,
    }


def run_hybrid_pipeline(
    model: nn.Module,
    test_samples: Sequence[Sample],
    out_dir: Path,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    device = device_name()
    model.eval()

    records = []

    for sample in test_samples:
        image = read_grayscale(sample.image)
        _, pred_mask = predict_mask(model, sample.image, device)

        df = region_feature_table(image, pred_mask)
        image_id = sample.image.stem

        n_objects = int(len(df))
        mean_area = float(df["area"].mean()) if not df.empty else 0.0
        mean_ecc = float(df["eccentricity"].mean()) if not df.empty else 0.0
        mean_solidity = float(df["solidity"].mean()) if not df.empty else 0.0
        foreground_fraction = float(pred_mask.mean())

        # Save raw quantitative feature table: auditable source data.
        df.to_csv(
            out_dir / f"{image_id}_unet_regionprops.csv",
            index=False,
        )

        prompt = HYBRID_JSON_PROMPT.format(
            image_id=image_id,
            n_objects=n_objects,
            mean_area=mean_area,
            foreground_fraction=foreground_fraction,
            mean_eccentricity=mean_ecc,
            mean_solidity=mean_solidity,
        )

        fallback = deterministic_hybrid_fallback(image_id, df, pred_mask)
        record = fallback
        llm_error = None

        try:
            text = ollama_chat(
                TEXT_MODEL,
                prompt,
                temperature=0.1,
                json_mode=True,
            )
            candidate = safe_json_loads(text)

            # Preserve measured numeric fields as source of truth.
            record = {
                "image_id": image_id,
                "n_objects": n_objects,
                "mean_area": round(mean_area, 2),
                "density_class": candidate.get(
                    "density_class", fallback["density_class"]
                ),
                "quality_flag": candidate.get(
                    "quality_flag", fallback["quality_flag"]
                ),
            }
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"

        narrative_prompt = NARRATIVE_PROMPT.format(
            record=json.dumps(record, ensure_ascii=False)
        )
        narrative = (
            f"Image {image_id} contained {record['n_objects']} segmented "
            f"objects with a mean area of {record['mean_area']:.2f} pixels. "
            f"The density was classified as {record['density_class']}, "
            f"and the quality flag was {record['quality_flag']}."
        )

        try:
            narrative = ollama_chat(
                TEXT_MODEL,
                narrative_prompt,
                temperature=0.15,
                json_mode=False,
            ).strip()
        except Exception:
            pass

        complete = {
            **record,
            "foreground_fraction": foreground_fraction,
            "mean_eccentricity": mean_ecc,
            "mean_solidity": mean_solidity,
            "narrative": narrative,
            "llm_error": llm_error,
        }

        save_json(
            {
                "measurements": {
                    "n_objects": n_objects,
                    "mean_area": mean_area,
                    "foreground_fraction": foreground_fraction,
                    "mean_eccentricity": mean_ecc,
                    "mean_solidity": mean_solidity,
                },
                "prompt": prompt,
                "structured_record": record,
                "narrative": narrative,
                "llm_error": llm_error,
            },
            out_dir / f"{image_id}_hybrid_record.json",
        )

        # Save input / U-Net mask visualization.
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("U-Net mask")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{image_id}_hybrid_mask.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

        records.append(complete)

    aggregate = pd.DataFrame(records)
    aggregate.to_csv(out_dir / "test_hybrid_records.csv", index=False)
    return aggregate


# Report-support summary


def write_report_support_summary(
    output_dir: Path,
    dataset_counts: Dict,
    task1_result: Dict,
    task2_result: Dict,
    comparison_df: pd.DataFrame,
    best_loss: str,
    best_metrics: Dict[str, float],
    otsu_metrics: Dict[str, float],
    hybrid_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Assignment 3 generated results summary\n")
    lines.append("This file is generated from actual pipeline outputs. "
                 "Use these values in the report rather than inventing results.\n")

    lines.append("## Dataset")
    for k, v in dataset_counts.items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Task 1")
    lines.append(f"- Optimised prompt saved in: task1/task1_vlm_results.json")
    if task1_result.get("ollama_error"):
        lines.append(f"- Ollama issue: {task1_result['ollama_error']}")
    else:
        lines.append(
            f"- Number of repeated optimised VLM runs: "
            f"{len(task1_result.get('optimized_runs', []))}"
        )

    lines.append("\n## Task 2")
    lines.append(f"- Feature summary: {task2_result.get('feature_summary')}")
    if task2_result.get("ollama_error"):
        lines.append(f"- Ollama issue: {task2_result['ollama_error']}")

    lines.append("\n## Task 3")
    lines.append(f"- Best U-Net loss: {best_loss}")
    lines.append(f"- Validation Dice: {best_metrics['dice']:.4f}")
    lines.append(f"- Validation IoU: {best_metrics['iou']:.4f}")
    lines.append(f"- Otsu validation Dice: {otsu_metrics['dice']:.4f}")
    lines.append(f"- Otsu validation IoU: {otsu_metrics['iou']:.4f}")

    if not comparison_df.empty:
        top = comparison_df.iloc[0]
        bottom = comparison_df.iloc[-1]
        lines.append(
            f"- Example where U-Net did better: {Path(top['image']).name}; "
            f"U-Net Dice={top['unet_dice']:.4f}, "
            f"Otsu Dice={top['otsu_dice']:.4f}"
        )
        lines.append(
            f"- Example where Otsu did better (or U-Net advantage was smallest): "
            f"{Path(bottom['image']).name}; "
            f"U-Net Dice={bottom['unet_dice']:.4f}, "
            f"Otsu Dice={bottom['otsu_dice']:.4f}"
        )

    lines.append("\n## Task 4")
    lines.append(f"- Test images processed: {len(hybrid_df)}")
    lines.append("- Aggregated CSV: task4/test_hybrid_records.csv")

    (output_dir / "REPORT_RESULTS_SUMMARY.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )



# Main pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Assignment 3 complete AI imaging pipeline"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="assignment3_run",
        help="Folder used for dataset and outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an already extracted dataset",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional path to already extracted dataset",
    )
    args = parser.parse_args()

    set_seed()

    workdir = Path(args.workdir).resolve()
    ensure_dir(workdir)

    outputs = ensure_dir(workdir / "outputs")

    if args.dataset_dir:
        dataset_root = Path(args.dataset_dir).resolve()
    else:
        zip_path = workdir / "nuclei_dataset.zip"
        dataset_root = workdir / "dataset"

        if not args.skip_download:
            download_dataset(zip_path)
            extract_dataset(zip_path, dataset_root)

    print(f"[Data] Discovering images under {dataset_root} ...")
    samples = pair_images_and_masks(dataset_root)

    # Images without masks can still be useful as unseen test images.
    with_masks = [s for s in samples if s.mask is not None]
    without_masks = [s for s in samples if s.mask is None]

    if len(with_masks) < 3:
        raise RuntimeError(
            "Fewer than 3 image/mask pairs were detected. "
            "Inspect the extracted folders and, if necessary, adjust "
            "looks_like_mask_path()/pair_images_and_masks()."
        )

    train, val, test = split_samples(with_masks)

    # Prefer genuinely unlabelled images as unseen test data when present.
    if without_masks:
        test = without_masks

    print(
        f"[Data] train={len(train)}, val={len(val)}, "
        f"test={len(test)}, unpaired={len(without_masks)}"
    )

    dataset_counts = {
        "all_detected_images": len(samples),
        "paired_images": len(with_masks),
        "unpaired_images": len(without_masks),
        "train": len(train),
        "validation": len(val),
        "test": len(test),
    }
    save_json(dataset_counts, outputs / "dataset_counts.json")

    
    # Task 1
    
    task1_dir = ensure_dir(outputs / "task1")
    make_eda(train, task1_dir)

    representative = train[0].image
    task1_result = task1_vlm(representative, task1_dir)

   
    # Task 2
    
    task2_dir = ensure_dir(outputs / "task2")
    task2_result = task2_classical(representative, task2_dir)

    
    # Task 3
   
    task3_dir = ensure_dir(outputs / "task3")

    histories = {}
    metrics_rows = []
    models = {}

    for loss_name in LOSSES_TO_RUN:
        model, hist_df, metrics = train_one_model(
            train,
            val,
            task3_dir,
            loss_name=loss_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        histories[loss_name] = hist_df
        models[loss_name] = model
        metrics_rows.append({
            "method": f"U-Net ({loss_name})",
            "dice": metrics["dice"],
            "iou": metrics["iou"],
        })

    otsu_metrics = evaluate_otsu_against_ground_truth(val)
    metrics_rows.append({
        "method": "Classical Otsu",
        "dice": otsu_metrics["dice"],
        "iou": otsu_metrics["iou"],
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(task3_dir / "evaluation_metrics.csv", index=False)

    plot_training_curves(histories, task3_dir)

    # Pick the U-Net with highest validation Dice.
    unet_rows = metrics_df[metrics_df["method"].str.startswith("U-Net")]
    best_idx = unet_rows["dice"].idxmax()
    best_method = metrics_df.loc[best_idx, "method"]
    best_loss = best_method.split("(")[1].rstrip(")")
    best_metrics = {
        "dice": float(metrics_df.loc[best_idx, "dice"]),
        "iou": float(metrics_df.loc[best_idx, "iou"]),
    }
    best_model = models[best_loss]

    torch.save(
        best_model.state_dict(),
        task3_dir / "best_unet_model.pt",
    )

    save_validation_panels(
        best_model,
        val,
        task3_dir / "validation_panels",
        n=3,
    )

    comparison_df = find_best_examples(
        best_model,
        val,
        task3_dir,
    )

    
    # Task 4
   
    task4_dir = ensure_dir(outputs / "task4")
    hybrid_df = run_hybrid_pipeline(
        best_model,
        test,
        task4_dir,
    )

    
    # Report support
    
    write_report_support_summary(
        outputs,
        dataset_counts,
        task1_result,
        task2_result,
        comparison_df,
        best_loss,
        best_metrics,
        otsu_metrics,
        hybrid_df,
    )

    print("\n============================================================")
    print("PIPELINE COMPLETE")
    print("============================================================")
    print(f"Outputs: {outputs}")
    print(f"Best loss: {best_loss}")
    print(f"Validation Dice: {best_metrics['dice']:.4f}")
    print(f"Validation IoU: {best_metrics['iou']:.4f}")
    print(f"Hybrid CSV: {task4_dir / 'test_hybrid_records.csv'}")
    print(f"Report summary: {outputs / 'REPORT_RESULTS_SUMMARY.md'}")
    print("============================================================")


if __name__ == "__main__":
    main()
