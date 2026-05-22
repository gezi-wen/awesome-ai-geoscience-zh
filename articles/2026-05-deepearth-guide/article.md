# DeepEarth 深度解析：AI 如何理解地球的时空规律

> 走进 Allen AI 联合 Stanford、ASU 打造的行星级 GeoAI 基础模型，理解地球观测 AI 的最前沿。

---

## 一句话概括

DeepEarth 是一个**自监督、多模态、时空 GeoAI 模型**。给它地球上任意一个位置 + 时间，它输出一个 192 维的特征向量，编码了那个时空点的环境状态。这个向量可以直接用于预测火灾风险、植被含水量、气候变化趋势等任务。

---

## 为什么遥感需要专门的 AI 模型？

普通深度学习处理的是「图像像素」，但地球观测数据有四个天然维度：

| 维度 | 含义 | 特点 |
|------|------|------|
| X (经度) | 东西方向 | -180° ~ +180° |
| Y (纬度) | 南北方向 | -90° ~ +90° |
| Z (海拔) | 高度 | -500m ~ 9000m |
| T (时间) | 时刻 | 任意历史/未来时点 |

**传统方法的局限**：把卫星图当普通图片处理，丢失了地理和时间的先验知识。同一片森林在春天和秋天的「图片」完全不同，但你知道它们本质上是同一个生态系统。

DeepEarth 把「这是地球」这个事实编码进了模型结构。

---

## 核心创新：Earth4D 时空编码器

DeepEarth 的心脏是 **Earth4D** —— 一个行星级 (X, Y, Z, T) 位置编码器。

### 原理

借鉴 NVIDIA Instant-NGP 的多分辨率哈希编码，Earth4D 把空间和时间离散成多个分辨率层级的哈希网格：

```
空间编码 (xyz): 24 级 × 2 特征 = 48维
    粗粒度 ──→ 细粒度
    32      ──→ 2^24 分辨率

时空编码 — 3 个 3D 投影，各 24 级 × 2 特征:
    xyt (纬度 + 经度 + 时间) = 48维 × 2 × 继续多个投影
    yzt (经度 + 海拔 + 时间) = 48维
    xzt (纬度 + 海拔 + 时间) = 48维
    合计: 144维

总输出: 48 + 144 = 192维
```

**关键洞察**：不把 4D 坐标放进一个 4D 哈希表（那不可行），而是用 4 个 3D 投影——xyz（纯空间）加上 xyt、yzt、xzt（三个时空切片）。空间编码捕获「这个地方是什么」，时空编码捕获「这里如何随时间变化」。

### 一句话代码

```python
from deepearth.encoders.xyzt.earth4d import Earth4D

world_model = Earth4D()
embeddings = world_model(
    (51.9976, -0.7416, 110, "1941-06-01 09:00 GMT"),   # 布莱切利园
    (40.4433, -79.9436, 270, "1985-01-15 10:00 ET"),   # CMU
    (46.2330, 6.0557, 430, "1989-03-12 10:00 CET"),    # CERN
    (45.5308, -73.6128, 63, "2026-02-04 11:00 ET"),    # Mila
)
# embeddings.shape: [4, 192] ← 可训练的时空特征
```

时间支持人类可读格式（自动解析时区）或归一化 [0, 1] 数值。

---

## 两种坐标系

DeepEarth 支持两种坐标表示：

### 1. ECEF（地心地固坐标系）— 默认

WGS84 椭球模型 → 把经纬度高程转换为以地心为原点的三维直角坐标。适合全球尺度任务，物理上精确。

### 2. Geographic（地理直接映射）

```
x = 纬度 (-90 ~ +90)
y = 经度 (-180 ~ +180)
z = 海拔 (m)
t = 时间 (归一化)
```

更适合区域研究。同纬度点共享 x 坐标，有助于生态知识的跨区域迁移。

---

## Learned Hash Probing：让碰撞变成特征

多分辨率哈希编码的经典问题是**哈希碰撞**——不同坐标被分到同一个哈希桶，损失空间精度。

DeepEarth 的解法：**Learned Hash Probing**。不是避免碰撞，而是学习如何处理碰撞。具体做法：

- 每个哈希桶计算 K 个候选探针（默认 32）
- 可学习的索引码本（512 条）选择最佳探针
- 探针权重的熵正则化防止退化

这贡献了 **23% 误差降低**和**99% 参数缩减**（相比相同精度下直接扩大哈希表）。

---

## 实际应用：Live Fuel Moisture Content 预测

DeepEarth 在 NSF I-GUIDE 项目中用于预测**植被活体含水量**（Live Fuel Moisture Content, LFMC）——衡量野火风险的关键指标。

在 LFMC 基准上：
- 使用 Earth4D 时空编码器的模型超越了 Galileo 等预训练基础模型
- 5M 参数达到之前 500M+ 参数模型的精度（99% 参数缩减）
- 地理坐标余弦校正进一步提升了 4%

**这意味着**：一个轻量模型，借助正确的时空先验，可以匹敌参数量百倍于己的通用模型。

---

## 技术栈一览

| 组件 | 技术 | 作用 |
|------|------|------|
| HashEncoder | 多分辨率哈希网格 | 连续坐标 → 离散特征 |
| Earth4D | 4×3D 投影编码 | 4D 时空 → 192维向量 |
| Learned Probing | 可学习探针 + 码本 | 碰撞消解 |
| AdaptiveRange | 数据驱动归一化 | 区域数据集优化 |
| Precompute | 哈希预计算 + CUDA | 训练加速 10× |

---

## 接下来学什么？

1. **跑通 demo.py** — 几分钟体验 Earth4D 编码器
2. **读 arXiv 论文** — [Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding](https://arxiv.org/abs/2603.07039)
3. **跑 Caravan 基准** — `encoders/xyzt/benchmarks/caravan/`，生态预测实战
4. **贡献文档** — DeepEarth 明确欢迎协作者，联系 lance@ecodash.ai
5. **探索其他应用** — 火灾模拟、洪水预测、碳排放估算

---

## 参考

- [DeepEarth GitHub](https://github.com/legel/deepearth)
- [arXiv:2603.07039](https://arxiv.org/abs/2603.07039) — 正式论文
- [2026 World Modeling Workshop](https://world-model-mila.github.io/) — Mila Quebec AI Institute
- [Earth4D 代码](https://github.com/legel/deepearth/tree/main/encoders/xyzt)
- [Instant-NGP](https://nvlabs.github.io/instant-ngp/) — NVIDIA 多分辨率哈希编码（技术源头）

---

*首次发布于 2026-05-20 · 掘金 / 知乎*
