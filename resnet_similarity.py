#!/usr/bin/env python3
"""
SERL ResNet-10 (多层级特征版) 图像相似度比较工具
专门针对视触觉(Tactile)图像进行优化：
1. 支持提取中间层 (Stage 2/3) 特征，保留更多纹理和形状细节
2. 对比多种 Pooling 策略 (Flatten, GAP, GMP, SpatialLearnedEmbeddings, SpatialSoftmax)
3. 计算每种方法从头到尾的完整耗时
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
    """支持返回中间层特征的 ResNetEncoder"""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    norm: str = "group"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False):
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (observations.astype(jnp.float32) / 255.0 - mean) / std

        conv = ft.partial(self.conv, use_bias=False, dtype=self.dtype, kernel_init=nn.initializers.kaiming_normal())
        norm = ft.partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        act = getattr(nn, self.act)

        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init")(x)
        x = norm(name="norm_init")(x)
        x = act(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

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
            outputs[f'stage_{i+1}'] = x

        return outputs

# =============================================================================
# 2. Pooling 方法定义
# =============================================================================

class SpatialLearnedEmbeddings(nn.Module):
    """可学习的空间嵌入池化"""
    height: int
    width: int
    channel: int
    num_features: int = 8
    
    @nn.compact
    def __call__(self, features):
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (self.height, self.width, self.channel, self.num_features),
        )
        
        batch_size = features.shape[0]
        # features: (B, H, W, C) -> (B, H, W, C, 1)
        # kernel: (H, W, C, num_features) -> (1, H, W, C, num_features)
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), 
            axis=(1, 2)
        )
        # 结果: (B, C, num_features) -> (B, C * num_features)
        features = jnp.reshape(features, [batch_size, -1])
        return features


class SpatialSoftmax(nn.Module):
    """空间 Softmax 池化，输出每个通道的期望坐标"""
    height: int
    width: int
    channel: int
    temperature: float = 1.0
    
    @nn.compact
    def __call__(self, features):
        batch_size = features.shape[0]
        
        # 创建坐标网格
        pos_x, pos_y = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, self.width),
            jnp.linspace(-1.0, 1.0, self.height)
        )
        pos_x = pos_x.reshape(-1)  # (H*W,)
        pos_y = pos_y.reshape(-1)  # (H*W,)
        
        # features: (B, H, W, C) -> (B, C, H*W)
        features = features.transpose(0, 3, 1, 2).reshape(
            batch_size, self.channel, self.height * self.width
        )
        
        # Softmax attention
        softmax_attention = nn.softmax(features / self.temperature)  # (B, C, H*W)
        
        # 计算期望坐标
        expected_x = jnp.sum(pos_x * softmax_attention, axis=2)  # (B, C)
        expected_y = jnp.sum(pos_y * softmax_attention, axis=2)  # (B, C)
        
        # 拼接: (B, 2*C)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)
        return expected_xy


# =============================================================================
# 3. 特征提取器类
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
            
        # 初始化模型结构
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
        
        # 预初始化 pooling 层的参数 (用于 SpatialLearnedEmbeddings)
        self._init_pooling_params()
        print("[ResNet] 模型就绪")

    def _init_pooling_params(self):
        """预初始化各 Stage 的 SpatialLearnedEmbeddings 参数"""
        self.sle_params = {}
        rng = jax.random.PRNGKey(42)
        
        # Stage 2: 16x16x128, Stage 3: 8x8x256, Stage 4: 4x4x512
        stage_configs = {
            'stage_2': (16, 16, 128),
            'stage_3': (8, 8, 256),
            'stage_4': (4, 4, 512),
        }
        
        for stage_name, (h, w, c) in stage_configs.items():
            sle = SpatialLearnedEmbeddings(height=h, width=w, channel=c, num_features=8)
            dummy_input = jnp.zeros((1, h, w, c))
            rng, key = jax.random.split(rng)
            self.sle_params[stage_name] = sle.init(key, dummy_input)

    def _load_image(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image_input = image_input[:, :, ::-1]
            img = Image.fromarray(image_input)
        else:
            img = image_input.convert("RGB")
            
        img = img.resize((128, 128), Image.BICUBIC)
        return np.array(img)

    def extract_all_stages(self, img_batch):
        """返回所有 Stage 的特征字典"""
        params = {'params': self.params}
        return self.apply_fn(params, img_batch, train=False)

    def _cosine_sim(self, v1, v2):
        v1_norm = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6)
        v2_norm = v2 / (jnp.linalg.norm(v2, axis=-1, keepdims=True) + 1e-6)
        return float(jnp.dot(v1_norm.flatten(), v2_norm.flatten()))

    def compute_similarity_with_timing(self, image1_path, image2_path):
        """计算所有方法的相似度，并记录每种方法的完整耗时"""
        
        results = {}
        stage_configs = {
            'stage_2': (16, 16, 128),
            'stage_3': (8, 8, 256),
            'stage_4': (4, 4, 512),
        }
        
        # 预加载图像（这部分时间会分摊到每个方法）
        img1_np = self._load_image(image1_path)
        img2_np = self._load_image(image2_path)
        
        print(f"\n{'='*100}")
        print(f"{'Stage':<10} | {'Method':<25} | {'Similarity':<12} | {'Time (ms)':<12} | {'Description'}")
        print(f"{'-'*100}")
        
        for stage_name in ['stage_2', 'stage_3', 'stage_4']:
            h, w, c = stage_configs[stage_name]
            resolution_str = f"{h}x{w}x{c}"
            
            # ==================== 方法 A: Flatten ====================
            t0 = time.perf_counter()
            img1 = jnp.array(img1_np)[None, ...]
            img2 = jnp.array(img2_np)[None, ...]
            feats1 = self.extract_all_stages(img1)
            feats2 = self.extract_all_stages(img2)
            f1, f2 = feats1[stage_name], feats2[stage_name]
            flat1 = f1.reshape(1, -1)
            flat2 = f2.reshape(1, -1)
            sim_flat = self._cosine_sim(flat1, flat2)
            t_flat = (time.perf_counter() - t0) * 1000
            
            print(f"{stage_name:<10} | {'Flatten (Spatial)':<25} | {sim_flat:.4f}       | {t_flat:.2f}        | 强空间敏感")
            
            # ==================== 方法 B: GAP ====================
            t0 = time.perf_counter()
            img1 = jnp.array(img1_np)[None, ...]
            img2 = jnp.array(img2_np)[None, ...]
            feats1 = self.extract_all_stages(img1)
            feats2 = self.extract_all_stages(img2)
            f1, f2 = feats1[stage_name], feats2[stage_name]
            gap1 = jnp.mean(f1, axis=(1, 2))
            gap2 = jnp.mean(f2, axis=(1, 2))
            sim_gap = self._cosine_sim(gap1, gap2)
            t_gap = (time.perf_counter() - t0) * 1000
            
            print(f"{stage_name:<10} | {'GAP (Avg)':<25} | {sim_gap:.4f}       | {t_gap:.2f}        | 忽略位置")
            
            # ==================== 方法 C: GMP ====================
            t0 = time.perf_counter()
            img1 = jnp.array(img1_np)[None, ...]
            img2 = jnp.array(img2_np)[None, ...]
            feats1 = self.extract_all_stages(img1)
            feats2 = self.extract_all_stages(img2)
            f1, f2 = feats1[stage_name], feats2[stage_name]
            gmp1 = jnp.max(f1, axis=(1, 2))
            gmp2 = jnp.max(f2, axis=(1, 2))
            sim_gmp = self._cosine_sim(gmp1, gmp2)
            t_gmp = (time.perf_counter() - t0) * 1000
            
            print(f"{stage_name:<10} | {'GMP (Max)':<25} | {sim_gmp:.4f}       | {t_gmp:.2f}        | 最强特征")
            
            # ==================== 方法 D: SpatialLearnedEmbeddings ====================
            t0 = time.perf_counter()
            img1 = jnp.array(img1_np)[None, ...]
            img2 = jnp.array(img2_np)[None, ...]
            feats1 = self.extract_all_stages(img1)
            feats2 = self.extract_all_stages(img2)
            f1, f2 = feats1[stage_name], feats2[stage_name]
            
            sle = SpatialLearnedEmbeddings(height=h, width=w, channel=c, num_features=8)
            sle_out1 = sle.apply(self.sle_params[stage_name], f1)
            sle_out2 = sle.apply(self.sle_params[stage_name], f2)
            sim_sle = self._cosine_sim(sle_out1, sle_out2)
            t_sle = (time.perf_counter() - t0) * 1000
            
            print(f"{stage_name:<10} | {'SpatialLearnedEmbed':<25} | {sim_sle:.4f}       | {t_sle:.2f}        | 可学习注意力")
            
            # ==================== 方法 E: SpatialSoftmax ====================
            t0 = time.perf_counter()
            img1 = jnp.array(img1_np)[None, ...]
            img2 = jnp.array(img2_np)[None, ...]
            feats1 = self.extract_all_stages(img1)
            feats2 = self.extract_all_stages(img2)
            f1, f2 = feats1[stage_name], feats2[stage_name]
            
            ssm = SpatialSoftmax(height=h, width=w, channel=c, temperature=1.0)
            # SpatialSoftmax 没有可学习参数，用空字典
            ssm_out1 = ssm.apply({}, f1)
            ssm_out2 = ssm.apply({}, f2)
            sim_ssm = self._cosine_sim(ssm_out1, ssm_out2)
            t_ssm = (time.perf_counter() - t0) * 1000
            
            print(f"{stage_name:<10} | {'SpatialSoftmax':<25} | {sim_ssm:.4f}       | {t_ssm:.2f}        | 期望坐标")
            
            print(f"{'-'*100}")
            
            results[stage_name] = {
                'flatten': {'sim': sim_flat, 'time_ms': t_flat},
                'gap': {'sim': sim_gap, 'time_ms': t_gap},
                'gmp': {'sim': sim_gmp, 'time_ms': t_gmp},
                'sle': {'sim': sim_sle, 'time_ms': t_sle},
                'ssm': {'sim': sim_ssm, 'time_ms': t_ssm},
            }
            
        return results


# conda activate serl
# python resnet_similarity.py


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SERL ResNet-10 触觉图像相似度分析")

    # 对称彩色图像
    # parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_133928_yimu_1_flip.jpg")
    # parser.add_argument("--image2", type=str, help="第二张图像路径", default="record_yimu_monitor/20251210_133928_yimu_2.jpg")

    # 对称彩色图像 (水平镜像)
    parser.add_argument("--image1", type=str, help="第一张图像路径", default="record_yimu_monitor/20251210_133928_yimu_1.jpg")
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
        
        results = extractor.compute_similarity_with_timing(args.image1, args.image2)
        
        print("\n[方法说明]")
        print("  Flatten:              直接展平，严格比较每个空间位置（最敏感）")
        print("  GAP:                  全局平均池化，忽略位置信息")
        print("  GMP:                  全局最大池化，只看最强响应")
        print("  SpatialLearnedEmbed:  8套可学习空间注意力权重（随机初始化）")
        print("  SpatialSoftmax:       输出每个通道的期望坐标 (x, y)")
        
        print("\n[触觉图像建议]")
        print("  推荐: Stage 2/3 + Flatten，对形变位置最敏感")
        print("  注意: SpatialLearnedEmbed 此处使用随机权重，仅供参考")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
