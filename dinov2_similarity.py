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
    DINOv2图像预处理（适用于触觉传感器图像）
    
    不使用 CenterCrop，直接 Resize 到目标尺寸，保留完整图像信息
    这样可以捕捉到图像任意位置的形变
    """
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
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
        image_tensor = self.transform(image).unsqueeze(0).to(self.device) #图像预处理
        
        # 模型前向传播，返回CLS token特征
        # 参考 dinov2/models/vision_transformer.py 第325-330行
        # forward() 默认返回 self.head(ret["x_norm_clstoken"])
        features = self.model(image_tensor)
        
        # L2归一化 - 对于相似度计算非常重要
        # 参考 dinov2/eval/utils.py 第21-27行 ModelWithNormalize
        features = F.normalize(features, dim=1, p=2)


        # with torch.no_grad():
        #     # 使用 forward_features 而不是 forward
        #     # 参考: vision_transformer.py 第255行
        #     output = self.model.forward_features(image_tensor)
    
        # # output 字典包含 (参考: vision_transformer.py 第265-270行):
        # # - "x_norm_clstoken": CLS token [1, embed_dim]
        # # - "x_norm_patchtokens": patch特征 [1, num_patches, embed_dim]
        # # - "x_norm_regtokens": register tokens (如果有)
        
        # patch_features = output["x_norm_patchtokens"]
        
        # # reshape 成空间形式 [1, H, W, embed_dim]
        # B, N, D = patch_features.shape
        # H = W = int(N ** 0.5)
        # spatial_features = patch_features.reshape(B, H, W, D)
            
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
        原有方法：使用 forward() 输出计算相似度（经过 head 层的 CLS token）
        
        Args:
            image1, image2: 图像输入
        
        Returns:
            相似度分数 (-1 到 1，越接近1越相似)
        """
        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)
        
        similarity = F.cosine_similarity(feat1, feat2).item()
        return similarity
    
    @torch.no_grad()
    def compute_similarity_cls(self, image1, image2) -> float:
        """
        仅使用 CLS Token 计算相似度
        
        CLS Token 是模型的全局特征表示，捕捉图像的整体语义信息
        参考: dinov2/models/vision_transformer.py 第255-271行 forward_features()
        
        Args:
            image1, image2: 图像输入
        
        Returns:
            相似度分数 (-1 到 1，越接近1越相似)
        """
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)
        tensor1 = self.transform(img1).unsqueeze(0).to(self.device)
        tensor2 = self.transform(img2).unsqueeze(0).to(self.device)
        
        # 使用 forward_features 获取原始特征（不经过 head 层）
        out1 = self.model.forward_features(tensor1)
        out2 = self.model.forward_features(tensor2)
        
        # 获取 CLS token: [1, embed_dim]
        cls_feat1 = out1["x_norm_clstoken"]
        cls_feat2 = out2["x_norm_clstoken"]
        
        # L2 归一化
        cls_feat1 = F.normalize(cls_feat1, dim=1, p=2)
        cls_feat2 = F.normalize(cls_feat2, dim=1, p=2)
        
        similarity = F.cosine_similarity(cls_feat1, cls_feat2).item()
        return similarity
    
    @torch.no_grad()
    def compute_similarity_patch(self, image1, image2) -> float:
        """
        仅使用 Patch Tokens 计算相似度（对所有 patch 取平均）
        
        Patch Tokens 保留空间信息，每个 token 对应图像的一个局部区域
        224x224 图像 / patch_size=14 → 16x16=256 个 patch tokens
        参考: dinov2/models/vision_transformer.py 第255-271行 forward_features()
        
        Args:
            image1, image2: 图像输入
        
        Returns:
            相似度分数 (-1 到 1，越接近1越相似)
        """
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)
        tensor1 = self.transform(img1).unsqueeze(0).to(self.device)
        tensor2 = self.transform(img2).unsqueeze(0).to(self.device)
        
        # 使用 forward_features 获取完整输出
        out1 = self.model.forward_features(tensor1)
        out2 = self.model.forward_features(tensor2)
        
        # 获取 patch tokens: [1, num_patches, embed_dim] → 取平均 → [1, embed_dim]
        patch_feat1 = out1["x_norm_patchtokens"].mean(dim=1)
        patch_feat2 = out2["x_norm_patchtokens"].mean(dim=1)
        
        # L2 归一化
        patch_feat1 = F.normalize(patch_feat1, dim=1, p=2)
        patch_feat2 = F.normalize(patch_feat2, dim=1, p=2)
        
        similarity = F.cosine_similarity(patch_feat1, patch_feat2).item()
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
# conda activate foundationpose
# python dinov2_similarity.py 
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DINOv2图像相似度比较")
    # 对称彩色图像
    parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_133928_yimu_1_flip.jpg")
    parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_133928_yimu_2.jpg")

    # parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_134705_yimu_1_flip.jpg")
    # parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_134705_yimu_2.jpg")
    # parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_141216_yimu_1_flip_gray.jpg")
    # parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_141216_yimu_2_gray.jpg")

    #不对称彩色图像
    # parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_141216_yimu_1_flip_color.jpg")
    # parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_141216_yimu_2_color.jpg")
    parser.add_argument("--model", type=str, 
                        default="dinov2_models/dinov2_vits14_pretrain.pth",
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
        
        # 计算三种相似度
        sim_original = extractor.compute_similarity(args.image1, args.image2)
        sim_cls = extractor.compute_similarity_cls(args.image1, args.image2)
        sim_patch = extractor.compute_similarity_patch(args.image1, args.image2)
        
        print("\n" + "=" * 60)
        print(f"图像1: {args.image1}")
        print(f"图像2: {args.image2}")
        print("=" * 60)
        print("\n【三种相似度计算方法对比】")
        print("-" * 60)
        print(f"方法1 - 原有方法 (forward+head): {sim_original:.4f}")
        print(f"方法2 - 仅 CLS Token:            {sim_cls:.4f}")
        print(f"方法3 - 仅 Patch Tokens (平均):  {sim_patch:.4f}")
        print("-" * 60)
        
        # 相似度解读（使用原有方法结果）
        print("\n【原有方法相似度解读】")
        if sim_original > 0.9:
            print(f"  {sim_original:.4f} → 高度相似 / 几乎相同")
        elif sim_original > 0.7:
            print(f"  {sim_original:.4f} → 相似 (可能是同类物体/场景)")
        elif sim_original > 0.5:
            print(f"  {sim_original:.4f} → 有一定相关性")
        else:
            print(f"  {sim_original:.4f} → 不太相似")
        
        print("\n【说明】")
        print("  - 原有方法: 使用 forward() 输出 (CLS token 经过 head 层)")
        print("  - CLS Token: 直接使用 forward_features() 的 CLS token")
        print("  - Patch Tokens: 所有 patch tokens 取平均，保留更多空间信息")
        print("=" * 60)

