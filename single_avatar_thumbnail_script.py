"""
single_avatar_thumbnail_script.py
────────────────────────────
Given
  • thumbnail_image_path_rgb  - path to an RGB still frame (PNG/JPG/…)
  • ckpt_path                 - path to the pre-trained MODNet weights (.pth)

The script
  1) estimates the alpha-matte with MODNet,
  2) composites the subject on white + transparent backgrounds,
  3) finds the square “half-body” crop (MediaPipe FaceMesh),
  4) writes four PNGs:

     <stem>_uncrop_white.png   - full frame, white background
     <stem>_uncrop_trans.png   - full frame, transparent BGRA
     <stem>_crop_white.png     - square crop, white background
     <stem>_crop_trans.png     - square crop, transparent BGRA
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms as T
import mediapipe as mp

from modnet_src.models.modnet import MODNet


# ─────────────────────────────────────────────────────────────────────────────
# 1. utilities – model loading & matte prediction
# ─────────────────────────────────────────────────────────────────────────────
_REF_SIZE = 512                                  # as used in the MODNet paper
_TO_TENSOR = T.Compose(
    [T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def _load_modnet(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    modnet = MODNet(backbone_pretrained=False)

    # <<<  wrap regardless of CPU/GPU  >>>
    modnet = torch.nn.DataParallel(modnet)      # <— move outside the if-block
    modnet.to(device)

    weights = torch.load(ckpt_path, map_location=device)
    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet



def _predict_matte(
    img_bgr: np.ndarray,
    modnet: torch.nn.Module,
    device: torch.device,
    ref_size: int = _REF_SIZE,
) -> np.ndarray:
    """
    Run MODNet on a single BGR image → return uint8 alpha matte (H×W, 0-255).
    """
    h, w = img_bgr.shape[:2]

    # unify channels to 3
    if img_bgr.ndim == 2:
        img_bgr = np.repeat(img_bgr[:, :, None], 3, axis=2)
    elif img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]

    # to tensor & normalize
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = _TO_TENSOR(img_rgb)            # C×H×W, float32
    img_tensor = img_tensor.unsqueeze(0)        # add batch dim

    # resize so that max/min side ≈ ref_size and multiples of 32
    _, _, im_h, im_w = img_tensor.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            rh, rw = ref_size, int(im_w / im_h * ref_size)
        else:
            rw, rh = ref_size, int(im_h / im_w * ref_size)
    else:
        rh, rw = im_h, im_w
    rh -= rh % 32
    rw -= rw % 32
    img_tensor = F.interpolate(img_tensor, size=(rh, rw), mode="area")

    # inference
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        _, _, matte = modnet(img_tensor, True)

    # back to original size
    matte = F.interpolate(matte, size=(h, w), mode="area")[0, 0].cpu().numpy()
    alpha_uint8 = (matte * 255).astype("uint8")
    return alpha_uint8


# ─────────────────────────────────────────────────────────────────────────────
# 2. main helper – produces the four PNGs
# ─────────────────────────────────────────────────────────────────────────────
def save_thumbnail_and_preview_images(
    thumbnail_image_path_rgb: str,
    ckpt_path: str,
    scale_factor: float = 3.0,
) -> Tuple[str, str, str, str]:
    """
    Returns absolute paths of the four saved PNGs.
    """
    # 2-a. load RGB frame
    frame_bgr = cv2.imread(thumbnail_image_path_rgb, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError(f"Could not read image: {thumbnail_image_path_rgb}")
    h, w, _ = frame_bgr.shape

    # 2-b. get MODNet matte
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modnet = _load_modnet(ckpt_path, device)
    alpha_uint8 = _predict_matte(frame_bgr, modnet, device)         # H×W

    alpha_norm = (alpha_uint8.astype(np.float32) / 255.0)[..., None]  # H×W×1

    # 2-c. build uncropped composites
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    white_bg   = np.full_like(frame_rgb, 255, dtype=np.float32)

    uncrop_white_rgb  = (frame_rgb * alpha_norm + white_bg * (1.0 - alpha_norm)).astype(np.uint8)
    uncrop_white_bgr  = cv2.cvtColor(uncrop_white_rgb, cv2.COLOR_RGB2BGR)

    uncrop_trans_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    uncrop_trans_bgra[:, :, 3] = alpha_uint8

    # 2-d. detect face & compute square half-body crop
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:
        res = face_mesh.process(frame_rgb.astype(np.uint8))
        if not res.multi_face_landmarks:
            raise ValueError("No face detected")

        lm = res.multi_face_landmarks[0].landmark
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        face_w, face_h = max(xs) - min(xs), max(ys) - min(ys)
        nose_x, nose_y = int(lm[1].x * w), int(lm[1].y * h)

        sq = int(max(face_w, face_h) * scale_factor)
        x0, y0 = nose_x - sq // 2, nose_y - sq // 2
        x0 = max(0, min(x0, w - sq))
        y0 = max(0, min(y0, h - sq))
        x1, y1 = x0 + sq, y0 + sq

    # 2-e. cropped composites
    crop_white_bgr  = uncrop_white_bgr[y0:y1, x0:x1]
    crop_trans_bgra = uncrop_trans_bgra[y0:y1, x0:x1]

    # 2-f. save
    stem = Path(thumbnail_image_path_rgb).with_suffix("")
    fn_uncrop_white = f"{stem}_uncrop_white.png"
    fn_uncrop_trans = f"{stem}_uncrop_trans.png"
    fn_crop_white   = f"{stem}_crop_white.png"
    fn_crop_trans   = f"{stem}_crop_trans.png"

    ok1 = cv2.imwrite(fn_uncrop_white, uncrop_white_bgr)
    ok2 = cv2.imwrite(fn_uncrop_trans, uncrop_trans_bgra)
    ok3 = cv2.imwrite(fn_crop_white,   crop_white_bgr)
    ok4 = cv2.imwrite(fn_crop_trans,   crop_trans_bgra)
    if not (ok1 and ok2 and ok3 and ok4):
        raise IOError("Could not write one or more thumbnails")

    return (
        os.path.abspath(fn_uncrop_white),
        os.path.abspath(fn_uncrop_trans),
        os.path.abspath(fn_crop_white),
        os.path.abspath(fn_crop_trans),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate avatar thumbnails and previews.")
    parser.add_argument("--thumbnail_image_path_rgb", type=str, required=True,
                        help="Path to the RGB thumbnail image (PNG/JPG/…).")
    parser.add_argument("--ckpt_path", type=str, default="modnet_photographic_portrait_matting.ckpt",
                        help="Path to the pre-trained MODNet weights (.pth).")
    parser.add_argument("--scale_factor", type=float, default=3.0,
                        help="Scale factor for the square half-body crop.")
    args = parser.parse_args()
    paths = save_thumbnail_and_preview_images(
        thumbnail_image_path_rgb=args.thumbnail_image_path_rgb,
        ckpt_path=args.ckpt_path,
        scale_factor=args.scale_factor,
    )
    print("Saved PNGs:")
    for p in paths:
        print(" •", p)