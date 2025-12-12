#!/usr/bin/env python3
"""
SERL ResNet-10 (多层级特征版) 图像相似度比较工具
专门针对视触觉(Tactile)图像进行优化：
1. 支持提取中间层 (Stage 2/3) 特征，保留更多纹理和形状细节
2. 对比多种 Pooling 策略 (Flatten, GAP, GMP)
"""

import sys
import os
import time
import pickle as pkl
import functools as ft
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import flax.linen as nn

# =============================================================================
# 1. 重新定义 ResNet 结构以支持提取中间层
#    (代码源自 serl_launcher，修改了 __call__ 返回值)
# =============================================================================

ModuleDef = Any

class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)

class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)

class ResNetEncoderWithIntermediates(nn.Module):
    """
    支持返回中间层特征的 ResNetEncoder
    """
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    norm: str = "group"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False):
        # 1. ImageNet 标准化
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (observations.astype(jnp.float32) / 255.0 - mean) / std

        # 2. 基础配置
        conv = ft.partial(self.conv, use_bias=False, dtype=self.dtype, kernel_init=nn.initializers.kaiming_normal())
        norm = ft.partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        act = getattr(nn, self.act)

        # 3. 初始卷积层
        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init")(x)
        x = norm(name="norm_init")(x)
        x = act(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # 4. 逐个 Stage 执行并收集输出
        outputs = {}
        
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)
            
            # 记录每个 Stage 结束后的特征图
            # Stage 编号从 1 开始
            outputs[f'stage_{i+1}'] = x

        return outputs

# =============================================================================
# 2. 特征提取器类
# =============================================================================

class ResNetFeatureExtractor:
    def __init__(self, model_path: str = None):
        # 查找权重
        if model_path is None:
            candidates = [
                "resnet10_params.pkl",
                os.path.expanduser("~/.serl/resnet10_params.pkl"),
                "serl/resnet10_params.pkl" 
            ]
            for p in candidates:
                if os.path.exists(p):
                    model_path = p
                    break
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError("未找到 resnet10_params.pkl，请检查路径。")
            
        print(f"[ResNet] 加载权重: {model_path}")
        with open(model_path, "rb") as f:
            self.params = pkl.load(f)
            
        # 初始化模型结构 (ResNet-10: stages=[1,1,1,1])
        self.model = ResNetEncoderWithIntermediates(
            stage_sizes=(1, 1, 1, 1), 
            block_cls=ResNetBlock
        )
        
        # 编译
        print("[ResNet] 正在编译 JAX 模型...")
        self.apply_fn = jax.jit(self.model.apply)
        
        # 预热
        dummy = jnp.zeros((1, 128, 128, 3), dtype=jnp.uint8)
        _ = self.extract_all_stages(dummy)
        print("[ResNet] 模型就绪")

    def _load_image(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image_input = image_input[:, :, ::-1] # BGR to RGB
            img = Image.fromarray(image_input)
        else:
            img = image_input.convert("RGB")
            
        img = img.resize((128, 128), Image.BICUBIC)
        return np.array(img)

    def extract_all_stages(self, img_batch):
        """返回所有 Stage 的特征字典"""
        params = {'params': self.params}
        return self.apply_fn(params, img_batch, train=False)

    def compute_similarity_analysis(self, image1_path, image2_path):
        # 1. 准备数据
        img1 = self._load_image(image1_path)[None, ...] # (1, 128, 128, 3)
        img2 = self._load_image(image2_path)[None, ...]
        
        # 2. 提取特征
        feats1 = self.extract_all_stages(img1)
        feats2 = self.extract_all_stages(img2)
        
        results = {}
        
        # 3. 逐层级、逐策略计算
        # Stage 1: 32x32 (过于低级，通常忽略)
        # Stage 2: 16x16, 128 ch (纹理、局部形状) -> 适合触觉
        # Stage 3: 8x8, 256 ch   (部件、中级语义) -> 适合触觉
        # Stage 4: 4x4, 512 ch   (全局语义)       -> 原版默认
        
        stages_to_test = ['stage_2', 'stage_3', 'stage_4']
        
        print(f"\n{'='*80}")
        print(f"{'Stage':<10} | {'Resolution':<12} | {'Method':<15} | {'Similarity':<10} | {'Description'}")
        print(f"{'-'*80}")

        for stage_name in stages_to_test:
            f1 = feats1[stage_name] # (1, H, W, C)
            f2 = feats2[stage_name]
            
            H, W, C = f1.shape[1:]
            resolution_str = f"{H}x{H}x{C}"
            
            # --- 策略 A: Flatten (保留空间结构) ---
            # 展平所有维度，严格比较每个像素点的特征
            flat1 = f1.reshape(1, -1)
            flat2 = f2.reshape(1, -1)
            sim_flat = self._cosine_sim(flat1, flat2)
            
            print(f"{stage_name:<10} | {resolution_str:<12} | {'Flatten(Spatial)':<15} | {sim_flat:.4f}     | 强空间敏感 (推荐触觉)")

            # --- 策略 B: GAP (Global Average Pooling) ---
            # 平均池化，忽略位置，只看特征是否存在
            gap1 = jnp.mean(f1, axis=(1, 2))
            gap2 = jnp.mean(f2, axis=(1, 2))
            sim_gap = self._cosine_sim(gap1, gap2)
            
            print(f"{stage_name:<10} | {resolution_str:<12} | {'GAP (Avg)':<15} | {sim_gap:.4f}     | 忽略位置，看整体")

            # --- 策略 C: GMP (Global Max Pooling) ---
            # 最大池化，只看最显著特征
            gmp1 = jnp.max(f1, axis=(1, 2))
            gmp2 = jnp.max(f2, axis=(1, 2))
            sim_gmp = self._cosine_sim(gmp1, gmp2)
            
            print(f"{stage_name:<10} | {resolution_str:<12} | {'GMP (Max)':<15} | {sim_gmp:.4f}     | 关注最强特征点")
            
            print(f"{'-'*80}")
            
            results[stage_name] = {
                'flatten': sim_flat, 
                'gap': sim_gap, 
                'gmp': sim_gmp
            }
            
        return results

    def _cosine_sim(self, v1, v2):
        # L2 归一化
        v1_norm = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6)
        v2_norm = v2 / (jnp.linalg.norm(v2, axis=-1, keepdims=True) + 1e-6)
        return jnp.dot(v1_norm, v2_norm.T).item()


# conda activate serl
# python resnet_similarity.py



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SERL ResNet-10 触觉图像相似度分析")

    # 对称彩色图像
    parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_133928_yimu_1_flip.jpg")
    parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_133928_yimu_2.jpg")


    # 不对称彩色图像 1
    # parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_141216_yimu_1_flip_color.jpg")
    # parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_141216_yimu_2_color.jpg")


    parser.add_argument("--model", type=str, default=None, help="resnet10_params.pkl 路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image1) or not os.path.exists(args.image2):
        print(f"错误: 找不到图像文件。\n1: {args.image1}\n2: {args.image2}")
        sys.exit(1)

    try:
        extractor = ResNetFeatureExtractor(model_path=args.model)
        
        print("\n[开始分析]")
        print(f"图像1: {args.image1}")
        print(f"图像2: {args.image2}")
        
        t0 = time.time()
        extractor.compute_similarity_analysis(args.image1, args.image2)
        print(f"\n总耗时: {(time.time()-t0)*1000:.1f}ms")
        
        print("\n[触觉图像相似度建议]")
        print("1. 推荐关注 Stage 2 或 Stage 3 的 'Flatten(Spatial)' 结果。")
        print("   - 分辨率适中 (16x16 或 8x8)，能保留形变位置信息。")
        print("   - Flatten 策略强行要求对应位置特征一致，对不对称形变最敏感。")
        print("2. 避免使用 Stage 4 或 GAP。")
        print("   - Stage 4 (4x4) 分辨率太低，丢失位置细节。")
        print("   - GAP 完全忽略位置信息，无法区分'左边压扁'和'右边压扁'。")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
