#!/usr/bin/env python3
"""
DINOv2 图像特征提取与相似度比较工具

代码来源说明：
- 模型构建方式参考自: dinov2/hub/backbones.py (第64-110行)
- vit_small函数定义: dinov2/models/vision_transformer.py (第341-352行)
- _make_dinov2_model调用方式: dinov2/hub/backbones.py (第42-53行)
"""

import sys
import os

# 添加dinov2路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dinov2"))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from dinov2.models import vision_transformer as vits


def load_dinov2_from_local(model_path: str, with_registers: bool = False):
    """
    从本地加载DINOv2模型 (支持 vit_small, vit_base, vit_large, vit_giant2)
    
    Args:
        model_path: 本地模型权重路径
        with_registers: 是否使用带registers的版本
    
    Returns:
        加载好权重的模型
    """
    # 根据文件名判断架构
    filename = os.path.basename(model_path).lower()
    if "vits" in filename:
        arch_fn = vits.vit_small
    elif "vitb" in filename:
        arch_fn = vits.vit_base
    elif "vitl" in filename:
        arch_fn = vits.vit_large
    elif "vitg" in filename:
        arch_fn = vits.vit_giant2
    else:
        print(f"[警告] 无法从文件名 '{filename}' 判断模型架构，默认使用 vit_small")
        arch_fn = vits.vit_small

    # 公共参数
    kwargs = dict(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        block_chunks=0,
    )

    if with_registers:
        # 参考 dinov2/hub/backbones.py 第102-109行
        kwargs.update(dict(
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        ))
    
    model = arch_fn(**kwargs)
    
    # 加载权重
    # 参考 dinov2/hub/backbones.py 第58-59行
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model


def get_transform():
    """
    DINOv2标准图像预处理
    
    参考 dinov2/data/transforms.py 中的 make_classification_eval_transform
    """
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DINOv2FeatureExtractor:
    """DINOv2特征提取器"""
    
    def __init__(self, model_path: str, device: str = "cuda", with_registers: bool = False):
        """
        初始化特征提取器
        
        Args:
            model_path: 模型权重路径
            device: 运行设备 ("cuda" 或 "cpu")
            with_registers: 是否使用带registers的版本
        """
        self.device = device
        self.with_registers = with_registers
        
        # 加载模型
        self.model = load_dinov2_from_local(model_path, with_registers)
        self.model = self.model.to(device)
        self.model.eval()
        
        # 预处理
        self.transform = get_transform()
        
        print(f"[DINOv2] 模型已加载: {model_path}")
        print(f"[DINOv2] 设备: {device}, with_registers: {with_registers}")
    
    def _load_image(self, image_input):
        """加载图像，支持路径、PIL图像、numpy数组"""
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # 假设是BGR格式(OpenCV)，转换为RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image_input = image_input[:, :, ::-1]  # BGR -> RGB
            return Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"不支持的图像类型: {type(image_input)}")
    
    @torch.no_grad()
    def extract_features(self, image_input):
        """
        提取单张图像的特征
        
        Args:
            image_input: 图像路径、PIL图像或numpy数组(BGR)
        
        Returns:
            归一化后的特征向量 [1, embed_dim]
        """
        image = self._load_image(image_input)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 模型前向传播，返回CLS token特征
        # 参考 dinov2/models/vision_transformer.py 第325-330行
        # forward() 默认返回 self.head(ret["x_norm_clstoken"])
        features = self.model(image_tensor)
        
        # L2归一化 - 对于相似度计算非常重要
        # 参考 dinov2/eval/utils.py 第21-27行 ModelWithNormalize
        features = F.normalize(features, dim=1, p=2)
        
        return features
    
    @torch.no_grad()
    def extract_features_batch(self, image_inputs: list):
        """
        批量提取图像特征
        
        Args:
            image_inputs: 图像列表
        
        Returns:
            归一化后的特征矩阵 [N, embed_dim]
        """
        tensors = []
        for img in image_inputs:
            image = self._load_image(img)
            tensors.append(self.transform(image))
        
        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch)
        features = F.normalize(features, dim=1, p=2)
        
        return features
    
    def compute_similarity(self, image1, image2) -> float:
        """
        计算两张图像的余弦相似度
        
        Args:
            image1, image2: 图像输入
        
        Returns:
            相似度分数 (-1 到 1，越接近1越相似)
        """
        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)
        
        similarity = F.cosine_similarity(feat1, feat2).item()
        return similarity
    
    def compute_similarity_matrix(self, images1: list, images2: list = None) -> np.ndarray:
        """
        计算两组图像之间的相似度矩阵
        
        Args:
            images1: 第一组图像
            images2: 第二组图像，如果为None则计算images1内部的相似度
        
        Returns:
            相似度矩阵 [len(images1), len(images2)]
        """
        feat1 = self.extract_features_batch(images1)
        
        if images2 is None:
            feat2 = feat1
        else:
            feat2 = self.extract_features_batch(images2)
        
        # 矩阵乘法计算所有pairs的余弦相似度
        similarity_matrix = torch.mm(feat1, feat2.T)
        
        return similarity_matrix.cpu().numpy()
    
    def find_most_similar(self, query_image, gallery_images: list):
        """
        在gallery中找到与query最相似的图像
        
        Args:
            query_image: 查询图像
            gallery_images: 候选图像列表
        
        Returns:
            (最相似图像索引, 相似度分数, 所有相似度分数)
        """
        query_feat = self.extract_features(query_image)
        gallery_feats = self.extract_features_batch(gallery_images)
        
        similarities = torch.mm(query_feat, gallery_feats.T).squeeze()
        
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        return best_idx, best_score, similarities.cpu().numpy()


# ==================== 使用示例 ====================
# python dinov2_similarity.py 
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DINOv2图像相似度比较")
    # parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_134705_yimu_1_flip.jpg")
    # parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_134705_yimu_2.jpg")
    parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_141216_yimu_1_flip_gray.jpg")
    parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_141216_yimu_2_gray.jpg")

    parser.add_argument("--model", type=str, 
                        default="dinov2_models/dinov2_vitl14_pretrain.pth",
                        help="模型权重路径")
    parser.add_argument("--with-registers", action="store_true",
                        help="使用带registers的模型版本")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 如果没有提供图像，运行演示
    if args.image1 is None or args.image2 is None:
      
        # 检查模型文件是否存在
        model_path = os.path.join(os.path.dirname(__file__), args.model)
        if not os.path.exists(model_path):
            print(f"\n[警告] 模型文件不存在: {model_path}")
            print("请确保已下载模型权重文件")
            sys.exit(1)
        
        # 创建提取器
        extractor = DINOv2FeatureExtractor(
            model_path=model_path,
            device=args.device if torch.cuda.is_available() else "cpu",
            with_registers=args.with_registers
        )
        print("\n[成功] 模型加载完成，可以开始使用！")
        
    else:
        # 正式使用
        model_path = os.path.join(os.path.dirname(__file__), args.model)
        
        # 如果使用registers版本，自动切换模型路径
        if args.with_registers and "reg4" not in args.model:
            model_path = model_path.replace("_pretrain.pth", "_reg4_pretrain.pth")
            print(f"[提示] 自动切换到registers版本: {model_path}")
        
        extractor = DINOv2FeatureExtractor(
            model_path=model_path,
            device=args.device if torch.cuda.is_available() else "cpu",
            with_registers=args.with_registers
        )
        
        # 计算相似度
        similarity = extractor.compute_similarity(args.image1, args.image2)
        
        print("\n" + "=" * 60)
        print(f"图像1: {args.image1}")
        print(f"图像2: {args.image2}")
        print(f"相似度: {similarity:.4f}")
        print("=" * 60)
        
        # 相似度解读
        if similarity > 0.9:
            print("解读: 高度相似 / 几乎相同")
        elif similarity > 0.7:
            print("解读: 相似 (可能是同类物体/场景)")
        elif similarity > 0.5:
            print("解读: 有一定相关性")
        else:
            print("解读: 不太相似")

