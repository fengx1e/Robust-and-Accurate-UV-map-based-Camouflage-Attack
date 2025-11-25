import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from utils.general import non_max_suppression, scale_coords, xywh2xyxy, box_iou
from utils.plots import plot_one_box, color_list


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _prepare_textures(textures, device):
    if textures is None:
        return None
    if isinstance(textures, np.ndarray):
        tex = torch.from_numpy(textures)
    elif isinstance(textures, torch.Tensor):
        tex = textures.detach().clone()
    else:
        raise TypeError("Unsupported texture type")
    if tex.dim() == 5:
        tex = tex.unsqueeze(0)
    return tex.to(device)


def _tensor_to_image(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu()
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    img = img.numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@torch.no_grad()
def _run_detector(model, img_tensor: torch.Tensor, device: torch.device, conf_thres: float, iou_thres: float):
    img = img_tensor.unsqueeze(0).to(device)
    img = img.float() / 255.0
    out, _ = model(img)
    preds = non_max_suppression(out, conf_thres, iou_thres)[0]
    if preds is None:
        return torch.empty((0, 6)), img_tensor
    preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], img_tensor.shape[1:]).round()
    return preds.cpu(), img_tensor


def _draw_predictions(image: np.ndarray, preds: torch.Tensor, names):
    img_plot = image[:, :, ::-1].copy()
    palette = color_list()
    if preds is not None and len(preds):
        for *xyxy, conf, cls in preds.tolist():
            label = f"{names[int(cls)]} {conf:.2f}"
            color = palette[int(cls) % len(palette)]
            plot_one_box(xyxy, img_plot, label=label, color=color, line_thickness=2)
    return img_plot


def _compute_asr(labels: torch.Tensor, preds: torch.Tensor, img_shape: Tuple[int, int]) -> Tuple[int, int]:
    if labels is None or labels.numel() == 0:
        return 0, 0

    labels = labels.detach().cpu()
    gt_classes = labels[:, 1].long()
    gt_boxes = xywh2xyxy(labels[:, 2:6])
    gt_boxes[:, [0, 2]] *= img_shape[1]
    gt_boxes[:, [1, 3]] *= img_shape[0]

    total = gt_boxes.shape[0]
    if preds is None or len(preds) == 0:
        return total, total

    pred_boxes = preds[:, :4]
    pred_classes = preds[:, 5].long()
    success = 0

    for gt_cls, gt_box in zip(gt_classes, gt_boxes):
        if len(pred_boxes) == 0:
            success += 1
            continue
        ious = box_iou(gt_box.unsqueeze(0), pred_boxes)[0]
        matches = (pred_classes == gt_cls) & (ious >= 0.5)
        if not matches.any():
            success += 1
    return success, total


@torch.no_grad()
def evaluate_camouflage_dataset(model,
                                dataset,
                                device,
                                names,
                                save_root,
                                clean_textures,
                                adv_textures,
                                conf_thres=0.25,
                                iou_thres=0.45,
                                logger=None):
    save_root = Path(save_root)
    clean_dir = save_root / "clean"
    camo_dir = save_root / "camouflage"
    _ensure_dir(clean_dir)
    _ensure_dir(camo_dir)

    model.eval()

    def run_pass(target_dir: Path, textures, compute_asr=False):
        dataset.set_textures(_prepare_textures(textures, dataset.device))
        success_total = 0
        object_total = 0
        for idx in range(len(dataset)):
            sample = dataset[idx]
            img_tensor = sample[0]
            labels = sample[4]
            path = sample[5]
            preds, _ = _run_detector(model, img_tensor, device, conf_thres, iou_thres)
            image_np = _tensor_to_image(img_tensor)
            vis = _draw_predictions(image_np, preds, names)
            out_path = target_dir / f"{Path(path).stem}.jpg"
            Image.fromarray(vis[:, :, ::-1]).save(out_path)
            if compute_asr:
                success, total = _compute_asr(labels, preds, image_np.shape[:2])
                success_total += success
                object_total += total
        return success_total, object_total

    if clean_textures is not None:
        run_pass(clean_dir, clean_textures, compute_asr=False)
    success, total = run_pass(camo_dir, adv_textures, compute_asr=True)
    asr = (success / total) if total > 0 else 0.0
    msg = f"Camouflage ASR: {asr * 100:.2f}% ({success}/{max(total,1)})"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return {"asr": asr, "success": success, "total": total}
