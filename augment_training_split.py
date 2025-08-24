import os
import shutil
import random
from glob import glob
from tqdm import tqdm

# Try Albumentations; if missing we fall back to Keras ImageDataGenerator
USE_ALBU = True
try:
    import albumentations as A
    from albumentations.core.composition import Compose
except Exception:
    USE_ALBU = False

from sklearn.model_selection import train_test_split
import cv2
import numpy as np

SOURCE_DIR = "DATASET"                # your original dataset (0-9, A-Z)
SPLIT_ROOT = "DATASET_SPLIT"          # where we store train/val split
TRAIN_DIR = os.path.join(SPLIT_ROOT, "train")
VAL_DIR   = os.path.join(SPLIT_ROOT, "val")
AUG_PER_IMAGE = 6                     # how many synthetic images to generate per original train image
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

random.seed(42)

def ensure_split():
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        print("✅ Using existing DATASET_SPLIT/train & val")
        return

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    print("Splitting original DATASET into train/val (80/20)…")

    for cls in sorted(os.listdir(SOURCE_DIR)):
      src_cls = os.path.join(SOURCE_DIR, cls)
      if not os.path.isdir(src_cls): 
          continue
      imgs = [f for f in os.listdir(src_cls) if f.lower().endswith(IMG_EXTS)]
      if len(imgs) == 0:
          continue
      tr, va = train_test_split(imgs, test_size=0.2, random_state=42, shuffle=True)

      os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
      os.makedirs(os.path.join(VAL_DIR, cls),   exist_ok=True)

      for f in tr:
          shutil.copy(os.path.join(src_cls, f), os.path.join(TRAIN_DIR, cls, f))
      for f in va:
          shutil.copy(os.path.join(src_cls, f), os.path.join(VAL_DIR, cls, f))

    print("✅ Split done.")

def get_aug():
    if USE_ALBU:
        print("Using Albumentations for augmentation.")
        return Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.4)
            ], p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.CLAHE(p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.25, rotate_limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.9),
            A.RandomShadow(p=0.3),
            A.RandomFog(p=0.15),
            A.RandomRain(p=0.15),
            A.CoarseDropout(max_holes=6, max_height=0.2, max_width=0.2, min_holes=1, fill_value=0, p=0.5)
        ])
    else:
        print("Albumentations not found. Falling back to simple OpenCV transforms.")
        # simple fallback: brightness/contrast/flip/rotate
        return None

def apply_fallback_aug(img):
    # random flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    # random rotation (-20..20)
    angle = random.uniform(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, random.uniform(0.85, 1.15))
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # random brightness/contrast
    alpha = random.uniform(0.7, 1.3)  # contrast
    beta  = random.uniform(-35, 35)   # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def main():
    ensure_split()
    aug = get_aug()

    classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    for cls in classes:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS)]
        print(f"Augmenting class '{cls}' ({len(imgs)} originals)…")

        for fname in tqdm(imgs, ncols=80):
            src_path = os.path.join(cls_dir, fname)
            img = cv2.imread(src_path)
            if img is None:
                continue

            base, ext = os.path.splitext(fname)
            for i in range(AUG_PER_IMAGE):
                if USE_ALBU and aug is not None:
                    aug_img = aug(image=img)["image"]
                else:
                    aug_img = apply_fallback_aug(img)

                # ensure 3-channel
                if aug_img.ndim == 2:
                    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR)

                out_name = f"{base}_aug{i}{ext}"
                cv2.imwrite(os.path.join(cls_dir, out_name), aug_img)

    print("✅ Augmentation complete. Re-train your model using DATASET_SPLIT/train & val.")

if __name__ == "__main__":
    main()
