#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np

try:
    import imageio.v3 as iio
except Exception:
    import imageio as iio


def imread_any(path: str) -> np.ndarray:
    return np.asarray(iio.imread(path))

def to_float01(img: np.ndarray) -> np.ndarray:
    # image conversion to float
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    if np.issubdtype(img.dtype, np.floating):
        out = img.astype(np.float32)
        if out.max() > 1.5:
            out = out / 255.0
        return np.clip(out, 0.0, 1.0)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    mx = np.iinfo(img.dtype).max
    return img.astype(np.float32) / float(mx)

def imwrite_jpg(path: str, rgb01: np.ndarray, quality: int = 95) -> None:
    img = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    try:
        iio.imwrite(path, img, quality=quality)
    except TypeError:
        iio.imwrite(path, img)

def split_bgr_glassplate(glass: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    H, W = glass.shape
    h = H // 3
    B = glass[0:h, :]
    G = glass[h:2*h, :]
    R = glass[2*h:3*h, :]
    return B, G, R

def is_image_file(name: str) -> bool:
    ext = os.path.splitext(name.lower())[1]
    return ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

# ----------------------------
# Robust scoring primitives
# ----------------------------

def gradient_mag(img: np.ndarray) -> np.ndarray:
    # gradient
    # gx
    gx = img[:, 2:] - img[:, :-2]
    gx = np.pad(gx, ((0, 0), (1, 1)), mode="edge")
    # gy
    gy = img[2:, :] - img[:-2, :]
    gy = np.pad(gy, ((1, 1), (0, 0)), mode="edge")
    return np.sqrt(gx * gx + gy * gy)

def center_crop(img: np.ndarray, crop_frac: float) -> np.ndarray:
    # during experimentation, I wanted to focus more on the central part of the image
    # and refrain from border distortion so implemented a center crop
    if crop_frac >= 1.0:
        return img
    H, W = img.shape
    nh = max(5, int(H * crop_frac))
    nw = max(5, int(W * crop_frac))
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    return img[y0:y0 + nh, x0:x0 + nw]

def overlapping_views(ref: np.ndarray, mov: np.ndarray, dy: int, dx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return overlapping views for ref and mov when mov is shifted by (dy, dx).
    No wrap-around. This is the key fix vs np.roll scoring.
    """
    H, W = ref.shape

    # region in ref
    y0r = max(0, dy)
    y1r = min(H, H + dy)
    x0r = max(0, dx)
    x1r = min(W, W + dx)

    y0m = max(0, -dy)
    y1m = min(H, H - dy)
    x0m = max(0, -dx)
    x1m = min(W, W - dx)

    return ref[y0r:y1r, x0r:x1r], mov[y0m:y1m, x0m:x1m]

def ncc_score(a: np.ndarray, b: np.ndarray) -> float:

    va = a.ravel()
    vb = b.ravel()
    va = va - va.mean()
    vb = vb - vb.mean()
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-10 or nb < 1e-10:
        return -1e18
    return float(np.dot(va, vb) / (na * nb))

def l2_score(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).ravel()
    return float(np.dot(d, d))


@dataclass(frozen=True)
class AlignResult:
    dy: int
    dx: int
    score: float


def exhaustive_align(
    ref: np.ndarray,
    mov: np.ndarray,
    max_disp: int = 15,
    metric: str = "ncc",
    crop_frac: float = 0.7,
    min_overlap: int = 80,
) -> AlignResult:
    """
    Exhaustive search over shifts in [-max_disp, max_disp]^2.
    Uses gradient magnitude + overlap-only scoring (robust).
    """
    # Preprocess for scoring
    ref_s = gradient_mag(center_crop(ref, crop_frac))
    mov_s = gradient_mag(center_crop(mov, crop_frac))

    best = AlignResult(0, 0, -1e18 if metric == "ncc" else 1e18)

    for dy in range(-max_disp, max_disp + 1):
        for dx in range(-max_disp, max_disp + 1):
            a, b = overlapping_views(ref_s, mov_s, dy, dx)
            if a.shape[0] < min_overlap or a.shape[1] < min_overlap:
                continue

            if metric == "ncc":
                s = ncc_score(a, b)
                if s > best.score:
                    best = AlignResult(dy, dx, s)
            else:
                s = l2_score(a, b)
                if s < best.score:
                    best = AlignResult(dy, dx, s)

    return best

def downsample2(img: np.ndarray) -> np.ndarray:
    # technique of downsampling by average pooling
    H, W = img.shape
    H2, W2 = H // 2, W // 2
    if H2 == 0 or W2 == 0:
        return img
    img = img[:2 * H2, :2 * W2]
    return 0.25 * (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2])

def pyramid_align(
    ref: np.ndarray,
    mov: np.ndarray,
    metric: str = "ncc",
    max_disp: int = 15,
    crop_frac: float = 0.7,
    min_size: int = 128,
    refine_radius: int = 6,
) -> AlignResult:

    # pyramid align first recurses till the images are small
    # then subsequently aligns the images and then scales back by 2
    H, W = ref.shape
    if min(H, W) <= min_size:
        return exhaustive_align(ref, mov, max_disp=max_disp, metric=metric, crop_frac=crop_frac)

    ref_small = downsample2(ref)
    mov_small = downsample2(mov)

    coarse = pyramid_align(
        ref_small, mov_small,
        metric=metric,
        max_disp=max_disp,
        crop_frac=crop_frac,
        min_size=min_size,
        refine_radius=refine_radius,
    )

    dy0, dx0 = 2 * coarse.dy, 2 * coarse.dx

    # refine around dy0, dx0 at this scale
    ref_s = gradient_mag(center_crop(ref, crop_frac))
    mov_s = gradient_mag(center_crop(mov, crop_frac))

    best = AlignResult(dy0, dx0, -1e18 if metric == "ncc" else 1e18)

    for dy in range(dy0 - refine_radius, dy0 + refine_radius + 1):
        for dx in range(dx0 - refine_radius, dx0 + refine_radius + 1):
            a, b = overlapping_views(ref_s, mov_s, dy, dx)
            if a.shape[0] < 80 or a.shape[1] < 80:
                continue

            if metric == "ncc":
                s = ncc_score(a, b)
                if s > best.score:
                    best = AlignResult(dy, dx, s)
            else:
                s = l2_score(a, b)
                if s < best.score:
                    best = AlignResult(dy, dx, s)

    return best

def apply_shift_noroll(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Apply shift without wrap-around by placing moved pixels into a blank canvas.
    (For visualization, wrap-around looks ugly; this avoids that.)
    """
    H, W = img.shape
    out = np.zeros_like(img)

    # destination
    y0d = max(0, dy)
    y1d = min(H, H + dy)
    x0d = max(0, dx)
    x1d = min(W, W + dx)

    # source
    y0s = max(0, -dy)
    y1s = min(H, H - dy)
    x0s = max(0, -dx)
    x1s = min(W, W - dx)

    out[y0d:y1d, x0d:x1d] = img[y0s:y1s, x0s:x1s]
    return out


def align_glassplate(
    glass_img: np.ndarray,
    metric: str = "ncc",
    use_pyramid: bool = True,
    max_disp: int = 15,
    crop_frac: float = 0.7,
    min_size: int = 128,
    refine_radius: int = 6,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    # returns shifts as tuples (dy,G) for example
    glass = to_float01(glass_img)
    B, G, R = split_bgr_glassplate(glass)

    if use_pyramid:
        resG = pyramid_align(B, G, metric=metric, max_disp=max_disp, crop_frac=crop_frac,
                             min_size=min_size, refine_radius=refine_radius)
        resR = pyramid_align(B, R, metric=metric, max_disp=max_disp, crop_frac=crop_frac,
                             min_size=min_size, refine_radius=refine_radius)
    else:
        resG = exhaustive_align(B, G, max_disp=max_disp, metric=metric, crop_frac=crop_frac)
        resR = exhaustive_align(B, R, max_disp=max_disp, metric=metric, crop_frac=crop_frac)
    # for a cleaner implementation
    G_aligned = apply_shift_noroll(G, resG.dy, resG.dx)
    R_aligned = apply_shift_noroll(R, resR.dy, resR.dx)

    rgb01 = np.stack([R_aligned, G_aligned, B], axis=-1)
    return rgb01, (resG.dy, resG.dx), (resR.dy, resR.dx)

def write_index_html(out_dir: str, rows: List[Dict[str, str]], title: str = "Alignment Results") -> None:
    html_path = os.path.join(out_dir, "index.html")
    lines = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'>")
    lines.append(f"<title>{title}</title>")
    lines.append("<style>")
    lines.append("body{font-family:system-ui,Arial,sans-serif;margin:24px;}")
    lines.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;}")
    lines.append(".card{border:1px solid #ddd;border-radius:12px;padding:12px;}")
    lines.append("img{width:100%;border-radius:10px;}")
    lines.append(".meta{color:#444;font-size:13px;margin-top:8px;line-height:1.35;}")
    lines.append("code{background:#f6f6f6;padding:2px 4px;border-radius:6px;}")
    lines.append("</style></head><body>")
    lines.append(f"<h1>{title}</h1>")
    lines.append("<div class='grid'>")
    for r in rows:
        lines.append("<div class='card'>")
        lines.append(f"<div><strong>{r['name']}</strong></div>")
        lines.append(f"<img src='{r['out_img']}' alt='{r['name']}'>")
        lines.append("<div class='meta'>")
        lines.append(f"Metric: <code>{r['metric']}</code><br>")
        lines.append(f"Pyramid: <code>{r['pyramid']}</code><br>")
        lines.append(f"G offset (dy,dx): <code>{r['g_off']}</code><br>")
        lines.append(f"R offset (dy,dx): <code>{r['r_off']}</code>")
        lines.append("</div></div>")
    lines.append("</div></body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser(description="Robust Prokudin-Gorskii alignment")
    ap.add_argument("--input", "-i", required=True, help="Input image path OR folder")
    ap.add_argument("--outdir", "-o", default="web", help="Output directory")
    ap.add_argument("--metric", choices=["ncc", "l2"], default="ncc", help="Alignment metric")
    ap.add_argument("--max-disp", type=int, default=15, help="Search radius at base level")
    ap.add_argument("--no-pyramid", action="store_true", help="Disable pyramid")
    ap.add_argument("--min-size", type=int, default=128, help="Stop pyramid when min(H,W) <= min_size")
    ap.add_argument("--refine-radius", type=int, default=6, help="Refine window radius per level")
    ap.add_argument("--crop-frac", type=float, default=0.7, help="Center crop fraction for scoring")
    ap.add_argument("--make-html", action="store_true", help="Create web/index.html gallery")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    in_path = args.input
    use_pyramid = not args.no_pyramid

    files: List[str] = []
    if os.path.isdir(in_path):
        for name in sorted(os.listdir(in_path)):
            if is_image_file(name):
                files.append(os.path.join(in_path, name))
        if not files:
            print("No image files found in:", in_path)
            sys.exit(1)
    else:
        if not os.path.isfile(in_path):
            print("Input not found:", in_path)
            sys.exit(1)
        files = [in_path]

    rows: List[Dict[str, str]] = []

    for fp in files:
        base = os.path.basename(fp)
        stem = os.path.splitext(base)[0]
        out_img_name = f"{stem}_aligned.jpg"
        out_img_path = os.path.join(args.outdir, out_img_name)

        glass = imread_any(fp)
        rgb01, g_off, r_off = align_glassplate(
            glass,
            metric=args.metric,
            use_pyramid=use_pyramid,
            max_disp=args.max_disp,
            crop_frac=args.crop_frac,
            min_size=args.min_size,
            refine_radius=args.refine_radius,
        )

        imwrite_jpg(out_img_path, rgb01)

        print(f"{base}")
        print(f"  G offset (dy, dx): {g_off}")
        print(f"  R offset (dy, dx): {r_off}")
        print(f"  saved: {out_img_path}")

        rows.append({
            "name": base,
            "out_img": out_img_name,
            "metric": args.metric,
            "pyramid": "yes" if use_pyramid else "no",
            "g_off": str(g_off),
            "r_off": str(r_off),
        })

    if args.make_html:
        write_index_html(args.outdir, rows, title="Prokudin-Gorskii Alignment Results")
        print("wrote:", os.path.join(args.outdir, "index.html"))


if __name__ == "__main__":
    main()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
