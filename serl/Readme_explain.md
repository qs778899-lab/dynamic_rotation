
# 项目结构:

1. serl_launcher (high-level RL policy):

    action维度自适应，由env如franka_env定义

    RL架构是分布式RL架构，learner node 和 actor node之间通信基于agentlace

2. robot_servers (bridge between high-level rl policy and low-level robot arm controller):

    通信方式: HTTP POST, 不依赖ROS环境，有利于RL库环境和ROS依赖的隔离

    ***_env.py(如franka_env.py): send http post

    ***_server.py(如franka_server.py): receive http post

3. 





# 真机训练流程(以franka arm为例):

    # Terminal 0: 启动机器人服务器 (必须先启动)
    python serl_robot_infra/robot_servers/franka_server.py \
        --robot_ip=172.16.0.2 \
        --gripper_type=Robotiq

    # Terminal 1: 启动 Learner (可以在 GPU 服务器上)
    bash run_learner.sh

    # Terminal 2: 启动 Actor (必须在能连接机器人的电脑上)
    bash run_actor.sh


# 框架特点:

    1. learner node 和 actor node可以放在不同的服务器，有利于加速模型训练

    2. example(franka sim)有state observation和image observation两种examples。state observation拥有所有状态特权信息, image observation是真实环境中的视觉输入引导。

    3. image encoder 基于Imagenet进行监督训练
    


# 基于 ResNet 的 Image Encoder 流程框架

## 数据流概览

```
输入: image (B, 128, 128, 3) uint8
                    ↓
        Step 1: ImageNet 标准化
                    ↓
        Step 2: ResNet-10 Backbone
                    ↓
        Step 3: Pooling Layer
                    ↓
        Step 4: Bottleneck (可选)
                    ↓
输出: 特征向量 (B, 256) 或 (B, 4096)
```

---

## Step 1: ImageNet 标准化

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
x = (x / 255.0 - mean) / std
```

- **输入**: `(B, 128, 128, 3)` uint8
- **输出**: `(B, 128, 128, 3)` float32，范围约 `[-2, 2]`

标准化的作用是：让你的图像的颜色分布与 ImageNet 的"平均自然图像"对齐，这样预训练的 ResNet 能更好地识别特征。

---

## Step 2: ResNet-10 Backbone (冻结权重)

```
Conv7x7 → GroupNorm → ReLU → MaxPool → Stage1 → Stage2 → Stage3 → Stage4
```

- **输出**: `(B, 4, 4, 512)` float32
- **语义**: 512 个不同的高层特征通道，每个通道 4×4 空间分辨率

### 尺寸变化流程（128×128 输入）

```
输入图像      (B, 128, 128, 3)
     ↓
Conv 7×7, stride=2, 64 filters
     ↓
             (B, 64, 64, 64)
     ↓
MaxPool 3×3, stride=2
     ↓
             (B, 32, 32, 64)     ← Stage 1 的输入
     ↓

注：Conv 7×7 用大卷积核捕捉低频信息（整体轮廓），stride=2 快速降低分辨率；
    MaxPool 在 3×3 区域内取最大值——此时的"值"是特征激活强度（某个卷积核对该位置的响应），
    不是原始像素值。取最大值 = 只保留"这片区域最强的特征响应"，忽略弱响应，增强平移不变性。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: 1个Block, 64 ch, stride=(1,1)
         不下采样，只做特征变换
     ↓

注：特征变换 = 通过两层 3×3 卷积 + 残差连接，让网络学习"在原特征基础上做非线性组合"，
    提取边缘、纹理等低级视觉特征，但不改变空间分辨率。

             (B, 32, 32, 64)     ← 边缘、纹理特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 2: 1个Block, 128 ch, stride=(2,2)
         下采样 2x，通道翻倍
     ↓
             (B, 16, 16, 128)    ← 局部模式特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 3: 1个Block, 256 ch, stride=(2,2)
         下采样 2x，通道翻倍
     ↓
             (B, 8, 8, 256)      ← 部件级特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 4: 1个Block, 512 ch, stride=(2,2)
         下采样 2x，通道翻倍
     ↓
             (B, 4, 4, 512)      ← 高层语义特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Step 3: Pooling Layer (网络层权重可在RL训练中学习优化)

- **输入**: 来自 Backbone 的特征图 `(B, 4, 4, 512)`
- **作用**: 将 4×4 的空间特征图转换为固定长度的向量
- **输出**: 取决于池化方法，如 `(B, 512)` 或 `(B, 4096)`

SERL 提供 4 种池化方法：avg、max、SpatialLearnedEmbeddings、SpatialSoftmax。
详见下方 [Pooling Layer (空间池化)](#pooling-layer-空间池化) 部分。

---



## Step 4: Bottleneck (降维)

```python
x = Dense(256)(x)
x = LayerNorm()(x)
x = tanh(x)
```

- **输出**: `(B, 256)`
- **语义**: 压缩后的图像表征，值域 `[-1, 1]`

---

## Pooling Layer (空间池化)

Backbone 输出是 `(B, 4, 4, 512)` 的特征图，需要转换为固定长度的向量。SERL 提供 4 种池化方法：

### 方法 A: avg (全局平均池化) - 最简单

```python
x = jnp.mean(x, axis=(-3, -2))  # (B, 4, 4, 512) → (B, 512)
```

- **含义**: 对 4×4 的每个通道取平均，得到 512 维向量
- **优点**: 简单高效
- **缺点**: 丢失空间位置信息

---

### 方法 B: max (全局最大池化)

```python
x = jnp.max(x, axis=(-3, -2))  # (B, 4, 4, 512) → (B, 512)
```

- **含义**: 对 4×4 的每个通道取最大值
- **优点**: 保留最显著特征
- **缺点**: 同样丢失空间信息

---

### 方法 C: SpatialLearnedEmbeddings (学习任务特定的空间注意力)

```python
class SpatialLearnedEmbeddings(nn.Module):
    height: int       # 4
    width: int        # 4
    channel: int      # 512
    num_features: int = 5  # 默认 5，但 SERL 用 8

    @nn.compact
    def __call__(self, features):
        # features: (B, 4, 4, 512)
        
        # 可学习的权重核 kernel: (4, 4, 512, num_features)
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (self.height, self.width, self.channel, self.num_features),
        )
        
        # 加权求和
        # features: (B, 4, 4, 512, 1)
        # kernel:   (1, 4, 4, 512, num_features)
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), 
            axis=(1, 2)
        )
        # 结果: (B, 512, num_features)
        
        features = jnp.reshape(features, [batch_size, -1])
        # 最终: (B, 512 * num_features) = (B, 4096) when num_features=8
        
        return features
```

**详细计算过程**:

| 步骤 | Shape | 说明 |
|------|-------|------|
| 输入 | `(B, 4, 4, 512)` | 特征图 |
| 扩展 | `(B, 4, 4, 512, 1)` | 增加最后一维 |
| kernel | `(1, 4, 4, 512, 8)` | 可学习参数 |
| 相乘 | `(B, 4, 4, 512, 8)` | 广播乘法 |
| 求和 | `(B, 512, 8)` | 在 H,W 维度求和 |
| 展平 | `(B, 4096)` | 512 × 8 = 4096 |

**含义**:
- Kernel 是一个"注意力模板"，每个 `(4, 4)` 的空间位置都有不同的权重
- 学习"哪些空间位置更重要"
- 相比 avg/max，**保留了空间信息**
- `num_features=8` 意味着学 8 套不同的空间注意力模式

---

### 方法 D: SpatialSoftmax (来自机器人学习领域)

```python
class SpatialSoftmax(nn.Module):
    # 输出: (B, 512 * 2) = (B, 1024)
    # 每个通道输出 (expected_x, expected_y) 2个坐标
```

- **含义**: 对每个特征通道，计算"期望空间位置"
- **输出**: 每个通道激活的"重心坐标" `(x, y)`
- **优点**: 专门设计用于机器人，捕捉物体位置
