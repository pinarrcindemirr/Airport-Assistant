"""
Image preprocessing pipeline for the CLIP-based vision component.

Covers every subtask the brief asks for (image loading, resizing,
normalisation, optional augmentation, conversion to tensors, batch loaders).
Image loading uses OpenCV (see load_image()); the resize/normalise/tensor
steps below still use PIL + plain PyTorch for now.

Design note on augmentation and training:
  CLIP is used here as a FROZEN, pretrained model (per the brief - "Pretrained
  models such as CLIP and Whisper should be used as frozen models and do not
  require training"). No gradient ever flows through it, so augmentation
  cannot improve CLIP itself the way it would for a model being trained from
  scratch. Augmentation is still implemented and exposed here for two
  legitimate reasons: (1) the brief's preprocessing subtasks explicitly ask
  for it with code and sample output, and (2) it provides extra, on-the-fly
  degraded views for ad-hoc robustness testing, on top of the three fixed
  conditions (angle/dark/blur) already baked into the generated dataset by
  generate_mockups.py.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance

# Official CLIP (ViT-B/32) normalisation constants (Radford et al., 2021).
# Using CLIP's own training-time statistics, rather than generic ImageNet
# ones, is what makes a frozen CLIP model behave correctly at inference.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_INPUT_SIZE = 224

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT_DIR / "data" / "images" / "image_labels.csv"


def load_image(path: str | Path) -> Image.Image:
    """
    Load an image file from disk using OpenCV, forcing RGB.

    OpenCV decodes images as BGR (blue-green-red) channel order by
    convention, NOT RGB - every other library in this pipeline (PIL,
    PyTorch, CLIP) expects RGB. Forgetting this conversion is a classic,
    silent bug: the image "loads fine" and is the right shape, but every
    colour is wrong, which quietly degrades a colour-sensitive model like
    CLIP without raising any error. cv2.cvtColor here is what makes that
    conversion explicit rather than accidental.

    The result is wrapped back into a PIL Image (rather than staying a raw
    NumPy array) so the rest of this module - augment(), and the resize/
    tensor step - can keep working exactly as before regardless of which
    library performed the initial read.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"OpenCV could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_and_crop(img: Image.Image, size: int = CLIP_INPUT_SIZE) -> Image.Image:
    """
    Resize the shorter side to `size` with bicubic interpolation, then
    centre-crop to size x size. This matches CLIP's own published
    preprocessing exactly, which matters because CLIP's positional/patch
    embeddings were trained on images prepared this way - a plain
    stretch-resize would distort the aspect ratio and quietly hurt accuracy.
    """
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def augment(img: Image.Image, angle_range: float = 10.0,
            brightness_range: tuple[float, float] = (0.75, 1.25)) -> Image.Image:
    """
    Illustrative augmentation: small random rotation + brightness jitter.
    Kept as a separate, optional step (not applied by default - see module
    docstring) rather than baked into build_tensor, so callers can compare
    augmented vs non-augmented tensors explicitly (used in the demo below).
    """
    import random
    angle = random.uniform(-angle_range, angle_range)
    out = img.rotate(angle, resample=Image.BICUBIC, fillcolor=img.getpixel((0, 0)))
    factor = random.uniform(*brightness_range)
    out = ImageEnhance.Brightness(out).enhance(factor)
    return out


def to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a normalised (C, H, W) float32 tensor, ready for
    CLIP's image encoder. Implemented directly with torch (no torchvision):
    pixel values are scaled to [0, 1], reordered to channel-first, then
    normalised with CLIP's own mean/std per channel.
    """
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).clone()
    arr = arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0  # (C, H, W)
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (arr - mean) / std


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Invert CLIP normalisation, for visual sanity-checking only (not used at inference)."""
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def preprocess(path: str | Path, use_augment: bool = False) -> torch.Tensor:
    """Full pipeline: load -> (optional augment) -> resize/crop -> normalised tensor."""
    img = load_image(path)
    if use_augment:
        img = augment(img)
    img = resize_and_crop(img)
    return to_tensor_normalized(img)


@dataclass
class ImageRecord:
    filename: str
    record_id: str
    category: str
    variant: str
    condition: str


class AirportImageDataset(Dataset):
    """
    Wraps image_labels.csv (produced by data/images/generate_mockups.py) so
    the full reference + query image set can be preprocessed and embedded in
    efficient batches, rather than one image at a time. No train/validation
    split is needed here since CLIP is frozen (see module docstring); this
    dataset is used purely for batched inference during evaluation.
    """

    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST,
                 project_root: str | Path = ROOT_DIR, use_augment: bool = False):
        self.project_root = Path(project_root)
        self.use_augment = use_augment
        with open(manifest_path, newline="", encoding="utf-8") as f:
            self.records = [ImageRecord(**row) for row in csv.DictReader(f)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ImageRecord]:
        rec = self.records[idx]
        tensor = preprocess(self.project_root / rec.filename, use_augment=self.use_augment)
        return tensor, rec


def _collate(batch):
    """Custom collate: stack tensors, keep metadata as a plain list (not stackable)."""
    tensors, records = zip(*batch)
    return torch.stack(tensors), list(records)


def build_dataloader(manifest_path: str | Path = DEFAULT_MANIFEST,
                      project_root: str | Path = ROOT_DIR,
                      batch_size: int = 8, use_augment: bool = False) -> DataLoader:
    """
    Batch loader over the full image set. num_workers=0 by default: this
    project targets a single-process Streamlit app and Windows development
    environments, where num_workers>0 requires an `if __name__ == "__main__"`
    guard and a working multiprocessing spawn setup - not worth the
    complexity for a dataset this size (80 images).
    """
    dataset = AirportImageDataset(manifest_path, project_root, use_augment=use_augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, collate_fn=_collate)


if __name__ == "__main__":
    # Demo / sanity check: python -m backend.image.preprocess
    loader = build_dataloader(batch_size=8)
    batch_tensors, batch_records = next(iter(loader))

    print(f"Dataset size: {len(loader.dataset)} images")
    print(f"Batch tensor shape: {tuple(batch_tensors.shape)}  dtype: {batch_tensors.dtype}")
    print(f"Value range after normalisation: [{batch_tensors.min():.3f}, {batch_tensors.max():.3f}]")
    print(f"First record in batch: {batch_records[0]}")

    # Save a before/after preview for the report (original vs preprocessed+denormalised).
    sample_path = loader.dataset.project_root / loader.dataset.records[0].filename
    original = load_image(sample_path)
    tensor = preprocess(sample_path)
    reconstructed = Image.fromarray(
        (denormalize(tensor).permute(1, 2, 0).numpy() * 255).astype("uint8")
    )

    preview = Image.new("RGB", (original.width + reconstructed.width, max(original.height, reconstructed.height)), "white")
    preview.paste(original, (0, 0))
    preview.paste(reconstructed, (original.width, 0))
    out_path = ROOT_DIR / "data" / "images" / "_preprocess_preview.png"
    preview.save(out_path)
    print(f"Before/after preview saved to {out_path}")