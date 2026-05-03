"""Shared augmentation pipeline for roof and corrosion segmentation.

Uses albumentations for GPU-efficient augmentation with support for:
- Geometric transforms (rotation, flip, scale — safe for overhead imagery)
- Photometric transforms (brightness, contrast, JPEG compression)
- GSD jitter (simulates different satellite resolutions)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_roof_augmentation(crop_size: int = 512, mode: str = "train") -> A.Compose:
    """Augmentation pipeline for Stage 1 (roof footprint segmentation).

    Args:
        crop_size: output tile size (pixels)
        mode: 'train' or 'val'
    """
    if mode == "val":
        return A.Compose(
            [
                A.Resize(crop_size, crop_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=45,
                border_mode=0, p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10, 50)),
                    A.ISONoise(),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.Defocus(radius=(1, 3)),
                ],
                p=0.2,
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5,
            ),
            A.CLAHE(p=0.2),
            A.ImageCompression(quality_lower=90, quality_upper=100, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_corrosion_augmentation(crop_size: int = 512, mode: str = "train") -> A.Compose:
    """Augmentation pipeline for Stage 2 (corrosion segmentation).

    Key differences from roof augmentation:
    - Full 360° rotation (overhead view has no canonical orientation)
    - Stronger JPEG compression (simulates satellite compression artifacts)
    - GSD jitter via random scale (simulates different satellite resolutions)
    - No hue shift (corrosion color is the primary signal)
    """
    if mode == "val":
        return A.Compose(
            [
                A.Resize(crop_size, crop_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=360, border_mode=0, p=0.8),  # full rotation OK for overhead
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,  # GSD jitter: ±20% simulates 25–60cm range
                rotate_limit=0,
                border_mode=0,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10, 50)),
                    A.ISONoise(),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                ],
                p=0.2,
            ),
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.15, hue=0.0, p=0.5,
            ),
            A.CLAHE(p=0.3),
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
