"""
Prithvi 遥感基础模型 —— NASA × IBM 联合出品
=============================================
演示如何使用 HuggingFace 上的 Prithvi 模型进行遥感图像分析。

Prithvi 是 NASA 和 IBM 联合训练的遥感基础模型（Foundation Model），
基于 ViT (Vision Transformer) 架构，在海量卫星数据上预训练。

安装:
  pip install transformers torch torchvision huggingface_hub pillow

模型地址: https://huggingface.co/ibm-nasa-geospatial
"""

import torch

print("=" * 60)
print("Prithvi · NASA/IBM 遥感基础模型")
print("=" * 60)

# ─── 1. 模型家族 ──────────────────────────────
print("""
📦 Prithvi 模型家族 (HuggingFace: ibm-nasa-geospatial)

┌──────────────────────────────┬─────────┬──────────────────┐
│ 模型                         │ 参数    │ 下载量           │
├──────────────────────────────┼─────────┼──────────────────┤
│ Prithvi-EO-1.0-100M          │ 100M    │ 2,123 · ⭐282   │
│ Prithvi-EO-2.0-300M          │ 300M    │ 13,568 · ⭐40   │
│ Prithvi-EO-2.0-600M          │ 600M    │ 2,707 · ⭐15    │
│ Prithvi-EO-2.0-300M-TL       │ 300M    │ 17,911 · ⭐9    │
│ Prithvi-EO-2.0-300M-BurnScars│ 300M    │ 2,962 · ⭐4     │
│ Prithvi-EO-2.0-tiny-TL       │ Tiny    │ 4,866 · ⭐9     │
└──────────────────────────────┴─────────┴──────────────────┘

1.0 = 初代 (最知名), 2.0 = 升级版 (更流行)
TL = Timm 格式 (推荐用于微调)
""")

# ─── 2. 核心概念 ──────────────────────────────
print("=" * 60)
print("什么是遥感基础模型？")
print("=" * 60)
print("""
传统做法: 每个遥感任务从头训练一个模型
   ↓ 费时 + 数据不够
基础模型: 在海量遥感数据上预训练 → 学到了通用的'遥感视觉'
   ↓
下游任务只需微调 (fine-tune) → 小数据也能达到高性能

Prithvi 的优势:
  • 预训练数据: NASA Harmonized Landsat Sentinel-2 (HLS)
  • 自监督学习: 掩码自编码器 (Masked Autoencoder, MAE)
  • 架构: Vision Transformer (ViT)
  • 多时相: 支持不同季节/年份的卫星图像对比
""")

# ─── 3. 使用示例（概念代码）─────────────────
print("=" * 60)
print("使用示例")
print("=" * 60)
print("""
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

# 加载模型（需要网络）
model_name = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 输入: Sentinel-2 多光谱图像 (HLS 格式)
# HLS 有 6 个波段: Blue, Green, Red, NIR, SWIR1, SWIR2
image = Image.open("satellite_image.tif")

# 预处理
inputs = processor(images=image, return_tensors="pt")

# 推理 → 通用遥感特征
with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state  # 图像特征
    cls_token = outputs.pooler_output     # 全局特征

print(f"Patch特征: {features.shape}")    # [1, N_patches, 1024]
print(f"全局特征: {cls_token.shape}")    # [1, 1024]

# 用于下游任务:
# 1. 分类: 在 cls_token 上加 Linear 层
# 2. 分割: 用 SegFormer 等解码器对接特征图
# 3. 变化检测: 对比两个时相的特征差异
""")

# ─── 4. 下游应用 ──────────────────────────────
print("=" * 60)
print("下游应用场景")
print("=" * 60)
print("""
NASA 和 IBM 基于 Prithvi 开发了多个专用模型:

🔥 野火检测 (Burn Scars)
  → Prithvi-EO-2.0-300M-BurnScars
  → 输入火灾前后卫星图 → 输出烧伤范围

🌾 作物分类 (Crop Classification)
  → 多时相 HLS 数据 + Prithvi → 识别作物类型

🌊 洪水映射 (Flood Mapping)
  → 输入洪水前后 Sentinel-1 SAR → 输出洪水范围

🌡️ 变化检测 (Change Detection)
  → 对比不同年份的同一地点 → 检测城市扩张/森林砍伐
""")

# ─── 5. 与 TorchGeo 联合使用 ──────────────────
print("=" * 60)
print("与 TorchGeo 联合使用")
print("=" * 60)
print("""
# TorchGeo 加载数据 + Prithvi 提取特征

from torchgeo.datasets import RESISC45

# 1. TorchGeo 加载遥感数据集
dataset = RESISC45(root="./data", download=True)

# 2. Prithvi 作为特征提取器
features_list = []
for sample in dataset:
    image = sample['image']  # PIL Image from TorchGeo
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = model(**inputs).pooler_output
    features_list.append(feats)

# 3. 特征 → 下游任务
# 用提取的特征训练一个简单的分类器
import numpy as np
X = torch.cat(features_list).numpy()  # [N, 1024]
y = np.array([s['label'] for s in dataset])

from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y)
print(f"Prithvi+SVM 准确率: {clf.score(X, y):.1%}")
""")

print("=" * 60)
print("✅ Prithvi 演示完成")
print("=" * 60)
print("""
🔗 资源:
  • Models: https://huggingface.co/ibm-nasa-geospatial
  • GitHub: https://github.com/NASA-IMPACT/hls-foundation-os
  • Paper: https://arxiv.org/abs/2310.18660
  • Tutorials: https://github.com/NASA-IMPACT/hls-foundation-os/tree/main/docs
""")
