#!/usr/bin/env python3
"""
像素级相似度比较工具

给定前后两帧 (prev_image, image)，先检测像素变化区域并生成
0/1 掩码 delta_image，然后比较两个 delta_image 的相似度。
"""

import argparse
import os
from typing import Dict, Tuple

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """以 BGR 格式加载图片。"""
    if image_path is None:
        raise ValueError("Image path is None")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return image


def _ensure_same_shape(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    """如果尺寸不同则将 image 调整到 reference 的尺寸。"""
    if reference.shape[:2] == image.shape[:2]:
        return image
    height, width = reference.shape[:2]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """转换为灰度图。"""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def build_delta_image(
    prev_image: np.ndarray,
    current_image: np.ndarray,
    threshold: int = 30,
    min_area: int = 5,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    根据 camera_reader.detect_pixel_changes 相同的思路构建 0/1 掩码。

    Returns:
        delta (np.ndarray): uint8, 值为 {0, 1}
        stats (dict): 包含差异统计信息
    """
    if prev_image is None or current_image is None:
        raise ValueError("输入图像不能为空")

    current_image = _ensure_same_shape(prev_image, current_image)
    gray_prev = _to_gray(prev_image)
    gray_curr = _to_gray(current_image)

    diff = cv2.absdiff(gray_prev, gray_curr)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if min_area > 0:
        filtered = np.zeros_like(thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
        thresh = filtered

    delta = (thresh > 0).astype(np.uint8)
    changed_pixels = int(delta.sum())
    total_pixels = delta.size
    change_percentage = (changed_pixels / total_pixels) * 100 if total_pixels else 0.0

    stats: Dict[str, float] = {
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percentage": change_percentage,
    }

    return delta, {"stats": stats, "diff": diff, "mask": thresh}


def _flatten_binary(image: np.ndarray) -> np.ndarray:
    """将 0/1 图像拉平成 float32 向量。"""
    return image.astype(np.float32).reshape(-1)


def cosine_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算余弦相似度。"""
    vec_a = _flatten_binary(mask_a)
    vec_b = _flatten_binary(mask_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def iou_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算 IoU 相似度。"""
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        return 1.0
    return intersection / union


def dice_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算 Dice 系数。"""
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    size_sum = float(mask_a.sum() + mask_b.sum())
    if size_sum == 0:
        return 1.0
    return (2.0 * intersection) / size_sum


SIMILARITY_FUNCS = {
    "cosine": cosine_similarity,
    "iou": iou_similarity,
    "dice": dice_similarity,
}


def compute_similarity(delta1: np.ndarray, delta2: np.ndarray, metric: str) -> float:
    """统一的相似度入口。"""
    if delta1.shape != delta2.shape:
        delta2 = cv2.resize(
            delta2.astype(np.uint8),
            (delta1.shape[1], delta1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    func = SIMILARITY_FUNCS[metric]
    return func(delta1, delta2)


def save_debug_images(save_dir: str, prefix: str, delta_info: Dict):
    """保存差异调试图像。"""
    os.makedirs(save_dir, exist_ok=True)
    diff_path = os.path.join(save_dir, f"{prefix}_diff.png")
    mask_path = os.path.join(save_dir, f"{prefix}_mask.png")
    delta_path = os.path.join(save_dir, f"{prefix}_delta.png")

    cv2.imwrite(diff_path, delta_info["diff"])
    cv2.imwrite(mask_path, delta_info["mask"])
    cv2.imwrite(delta_path, (delta_info["mask"] > 0).astype(np.uint8) * 255)


# python pixel_similarity.py 
def main():
    parser = argparse.ArgumentParser(description="基于像素变化的相似度比较")
    parser.add_argument("--prev-image1", help="第一组上一帧", default="record_yimu_monitor/20251210_153718_yimu_1_flip.jpg")
    parser.add_argument("--prev-image2", help="第二组上一帧", default="record_yimu_monitor/20251210_153718_yimu_2.jpg")
    # parser.add_argument("--image1", help="第一组当前帧", default="record_yimu_monitor/20251210_141216_yimu_1_flip_color.jpg")
    # parser.add_argument("--image2", help="第二组当前帧", default="record_yimu_monitor/20251210_141216_yimu_2_color.jpg")
    # parser.add_argument("--image1", help="第一组当前帧", default="record_yimu_monitor/20251210_134705_yimu_1_flip.jpg")
    # parser.add_argument("--image2", help="第二组当前帧", default="record_yimu_monitor/20251210_134705_yimu_2.jpg")
    parser.add_argument("--image1", help="第一组当前帧", default="record_yimu_monitor/20251210_171301_yimu_1_flip.jpg")
    parser.add_argument("--image2", help="第二组当前帧", default="record_yimu_monitor/20251210_171301_yimu_2.jpg")
    parser.add_argument("--threshold", type=int, default=2, help="像素差异阈值")
    parser.add_argument("--min-area", type=int, default=8, help="最小变化区域面积")
    parser.add_argument(
        "--metric",
        choices=list(SIMILARITY_FUNCS.keys()),
        default="cosine",
        help="相似度度量方式",
    )
    parser.add_argument("--save-dir", type=str, default="record_yimu_monitor/", help="可选，调试图保存目录")
    args = parser.parse_args()

    prev_image1 = load_image(args.prev_image1)
    image1 = load_image(args.image1)
    prev_image2 = load_image(args.prev_image2)
    image2 = load_image(args.image2)

    delta1, debug1 = build_delta_image(
        prev_image1, image1, threshold=args.threshold, min_area=args.min_area
    )
    delta2, debug2 = build_delta_image(
        prev_image2, image2, threshold=args.threshold, min_area=args.min_area
    )

    similarity = compute_similarity(delta1, delta2, metric=args.metric)

    print("=" * 60)
    print("第一组变化统计:", debug1["stats"])
    print("第二组变化统计:", debug2["stats"])
    print(f"相似度 ({args.metric}): {similarity:.4f}")
    print("=" * 60)

    if args.save_dir:
        save_debug_images(args.save_dir, "pair1", debug1)
        save_debug_images(args.save_dir, "pair2", debug2)
        np.save(os.path.join(args.save_dir, "delta1.npy"), delta1)
        np.save(os.path.join(args.save_dir, "delta2.npy"), delta2)
        print(f"调试图及 delta 掩码已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()

