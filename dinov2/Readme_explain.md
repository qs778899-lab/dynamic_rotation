# DinoVisionTransformer 架构

## 1. 图像预处理流程

```
原始图像文件 (photo.jpg)
        ↓ Image.open()
PIL Image (W=640, H=480)
        ↓ Resize((224, 224))
PIL Image (W=224, H=224)
        ↓ ToTensor()
Tensor [3, 224, 224]  ← [C, H, W] (C: RGB通道, H: 高度, W: 宽度)
        ↓ unsqueeze(0)
Tensor [1, 3, 224, 224]  ← [B, C, H, W] (B: batch, 一次处理的图片数量)
```

---

## 2. 模型架构

### 整体流程

```
输入: [B, 3, H, W]
        ↓
    patch_embed (PatchEmbed)
    图像切片 + 线性投影
    [B, 3, H, W] → [B, num_patches, embed_dim]
        ↓
    + cls_token    添加 CLS token
    + pos_embed    添加位置编码
    + reg_tokens   添加 register tokens (可选)
        ↓
    blocks[0] (Attention + FFN)  Transformer Block 0
        ↓
    blocks[1]                    Transformer Block 1
        ↓
       ...                       (ViT-S: 12层, ViT-L: 24层, ViT-g: 40层)
        ↓
    blocks[N-1]                  Transformer Block N-1
        ↓
    norm (LayerNorm)
        ↓
    head (默认 Identity)
        ↓
输出: CLS token [B, embed_dim]
```

### forward_features() 返回值

| 键名 | 形状 | 物理含义与作用 |
|------|------|----------------|
| `x_norm_clstoken` | `[B, embed_dim]` | **全局特征**：整张图像的"摘要"，经过所有 Transformer 层后，CLS token 已经"见过"所有 patch，包含图像的全局语义信息。常用于图像分类、相似度比较。 |
| `x_norm_patchtokens` | `[B, 256, embed_dim]` | **局部特征**：256 个 patch token，每个对应图像的一个 14×14 区域。保留空间位置信息，可 reshape 成 16×16 的特征图。适用于目标检测、语义分割、局部形变检测。 |
| `x_norm_regtokens` | `[B, num_reg, embed_dim]` | **辅助 token**：Register tokens（默认 4 个），用于吸收特征图中的"伪影/噪声"，使 patch tokens 更干净。本身不携带语义信息，一般不直接使用。 |
| `x_prenorm` | `[B, 257+num_reg, embed_dim]` | **原始特征**：LayerNorm 之前的特征，包含所有 token。通常使用归一化后的版本（上面三个）。 |
| `masks` | - | **掩码**：用于 masked image modeling 训练，推理时通常为 None。 |


---

## 3. Patch 切分示意

### 原始图像 224×224

```
+-----+-----+-----+-----+
| P0  | P1  | P2  | ... |  ← 每个 patch 14×14 像素
+-----+-----+-----+-----+
| P16 | P17 | P18 | ... |
+-----+-----+-----+-----+
| ... | ... | ... | ... |    总共 16×16 = 256 个 patch
+-----+-----+-----+-----+
| ... | ... | ... |P255 |
+-----+-----+-----+-----+
```

### patch_embed 后

256 个 patch token，每个是 embed_dim 维向量（ViT-L 为 1024 维）:

```
P0   = [0.12, -0.34, 0.56, ..., 0.78]  ← 1024个数
P1   = [0.23, 0.45, -0.67, ..., 0.89]
...
P255 = [0.11, -0.22, 0.33, ..., 0.44]
```

### 添加 CLS token

```
CLS = [0.01, 0.02, 0.03, ..., 0.99]  ← 可学习参数，随机初始化
```

完整序列（257 个 token）:

```
[CLS, P0, P1, P2, ..., P255]
  ↑      ↑────────────────↑
 1个          256个
```

---

## 4. Self-Attention 机制

### 初始状态

```
[CLS] [P0] [P1] [P2] ... [P255]
```

### 经过 Self-Attention（所有 token 互相交流信息）

```
[CLS'] [P0'] [P1'] [P2'] ... [P255']
```

**CLS' 现在"见过"了所有 patch 的信息，变成了整张图的"摘要"**

---

## 5. 模型规格对比

| 模型 | 层数 | embed_dim | 参数量 |
|------|------|-----------|--------|
| ViT-S/14 | 12 | 384 | 21M |
| ViT-B/14 | 12 | 768 | 86M |
| ViT-L/14 | 24 | 1024 | 300M |
| ViT-g/14 | 40 | 1536 | 1.1B |
