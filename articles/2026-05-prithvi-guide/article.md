# 认识 Prithvi：NASA × IBM 的遥感基础模型

> HuggingFace 上最受欢迎的遥感 AI 模型。读完你会用 Prithvi 提取卫星图像特征，用于分类、分割、变化检测。

---

## Prithvi 是什么？

**Prithvi**（梵语「地球」）是 NASA 和 IBM 联合训练的遥感基础模型（Foundation Model）。2023 年发布后迅速成为 HuggingFace 上最受欢迎的遥感模型——初代 100M 版本获 282 次点赞，2.0 300M 版本被下载 13,000+ 次。

**核心思路**：在海量卫星数据上预训练一个 Vision Transformer → 学到通用的「遥感视觉」→ 任何下游任务只需微调。

---

## 模型家族

| 模型 | 参数 | 下载量 | 用途 |
|------|------|--------|------|
| **Prithvi-EO-1.0-100M** | 100M | 2,123 · ⭐282 | 初代经典，社区最认可 |
| **Prithvi-EO-2.0-300M** | 300M | 13,568 · ⭐40 | 升级版，下载量最高 |
| Prithvi-EO-2.0-600M | 600M | 2,707 · ⭐15 | 大杯，精度最高 |
| Prithvi-EO-2.0-300M-TL | 300M | 17,911 | Timm 格式，推荐用于微调 |
| Prithvi-EO-2.0-300M-BurnScars | 300M | 2,962 | 火灾烧伤检测 |
| Prithvi-EO-2.0-tiny-TL | Tiny | 4,866 | 轻量版，适合快速实验 |

TL 后缀 = Timm 库格式，社区推荐用于下游微调任务。

---

## 技术特点

### 训练方式

- **数据**：NASA HLS（Harmonized Landsat Sentinel-2），覆盖全球的多时相卫星图像
- **方法**：掩码自编码器（Masked Autoencoder, MAE）——随机遮挡图像中的 patch，训练模型从剩余部分重建被遮挡内容
- **架构**：Vision Transformer (ViT)

### 多光谱支持

Prithvi 不像普通 ViT 只能处理 RGB。它原生支持 **HLS 的 6 个光谱波段**：

| 波段 | 波长 | 用途 |
|------|------|------|
| Blue | 0.49 μm | 水体、大气 |
| Green | 0.56 μm | 植被 |
| Red | 0.66 μm | 植被健康 |
| NIR | 0.83 μm | **植被核心指标** |
| SWIR1 | 1.61 μm | 水分、矿物 |
| SWIR2 | 2.20 μm | 地质、土壤 |

这比 RGB 多出整整一个数量级的信息——尤其是 NIR（近红外），是计算 NDVI 等植被指数的关键波段。

### 多时相

Prithvi 能处理不同时间拍摄的同一地点图像。这让它天然适合：
- 变化检测（城市扩张、森林砍伐）
- 灾害响应（火灾前后、洪水前后）
- 作物物候分析（不同生长阶段）

---

## 快速上手

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

model_name = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 加载卫星图（HLS 6波段格式）
image = Image.open("satellite.tif")
inputs = processor(images=image, return_tensors="pt")

# 提取特征
with torch.no_grad():
    outputs = model(**inputs)
    global_feature = outputs.pooler_output  # [1, 1024]
    patch_features = outputs.last_hidden_state  # [1, N_patches, 1024]
```

1024 维的 `global_feature` 就是这张卫星图的「通用遥感表示」，可以直接输入任何下游分类器。

---

## 与 TorchGeo 互补使用

TorchGeo 负责数据加载，Prithvi 负责特征提取：

```python
from torchgeo.datasets import RESISC45
from transformers import AutoModel, AutoImageProcessor

# 数据
dataset = RESISC45(root="./data", download=True)

# 特征提取
features = []
for sample in dataset:
    inputs = processor(images=sample['image'], return_tensors="pt")
    with torch.no_grad():
        feat = model(**inputs).pooler_output
    features.append(feat)

# 下游分类
from sklearn.svm import SVC
clf = SVC().fit(torch.cat(features), labels)
```

TorchGeo 提供标准化数据接口，Prithvi 提供预训练特征——配合使用效果最佳。

---

## Prithvi 在技术栈中的位置

```
数据层 (TorchGeo)          → 加载 + 预处理
    ↓
特征层 (Prithvi)           → 通用遥感特征提取
    ↓
时空层 (DeepEarth)         → 加入时间/空间编码
    ↓
知识层 (EVE)               → RAG 检索相关文献
    ↓
写作层 (SciAssistant)      → 自动生成分析报告
```

---

## 参考

- [HuggingFace 模型](https://huggingface.co/ibm-nasa-geospatial)
- [GitHub 仓库](https://github.com/NASA-IMPACT/hls-foundation-os)
- [论文](https://arxiv.org/abs/2310.18660)
- [HLS 数据介绍](https://hls.gsfc.nasa.gov/)

---

*首次发布于 2026-05-20 · 掘金 / 知乎*
