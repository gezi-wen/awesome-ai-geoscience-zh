# TorchGeo 入门：用 PyTorch 处理遥感数据

> 面向 PyTorch 用户的遥感深度学习第一课。读完你将能用 TorchGeo 加载卫星图像、训练分类模型。

---

## 为什么需要 TorchGeo？

PyTorch 的 `torchvision` 处理自然图像很顺手，但遇到遥感数据就力不从心了：

- **地理坐标系** — 图片不是 `.png`，而是带投影的 GeoTIFF
- **多光谱通道** — 不止 RGB，还有近红外、热红外等十几个波段
- **超大尺寸** — 一张遥感图动辄上万像素，不能直接塞进 GPU
- **空间采样** — 不能随机 crop，要考虑地理重叠和空间自相关

TorchGeo 是 PyTorch 官方生态项目（微软开发），专门解决这些问题。它提供：

- 50+ 遥感数据集，一行代码下载
- 地理感知的采样器（随机/网格/预切分）
- 与 `torchvision` 和 PyTorch Lightning 无缝对接

一行安装：

```bash
pip install torchgeo
```

---

## 第一个遥感数据集

TorchGeo 内置了大量经典遥感数据集。以 **EuroSAT** 为例——27000 张 Sentinel-2 卫星图像，分为 10 种土地覆盖/利用类型：

```python
from torchgeo.datasets import EuroSAT

dataset = EuroSAT(root="./data", download=True)

print(len(dataset))          # 27000
print(dataset.num_classes)   # 10
print(dataset.classes)       # ['AnnualCrop', 'Forest', 'HerbaceousVegetation',
                             #  'Highway', 'Industrial', 'Pasture',
                             #  'PermanentCrop', 'Residential', 'River', 'SeaLake']
```

TorchGeo 数据集返回 dict，包含 `image`（PIL 或 Tensor）和 `label`（整数）：

```python
sample = dataset[0]
print(sample['image'].shape)   # torch.Size([3, 64, 64])
print(sample['label'])         # 0 → AnnualCrop
```

其他常用数据集（同样一行加载）：

| 数据集 | 任务 | 规模 | 类别 |
|--------|------|------|------|
| RESISC45 | 场景分类 | 31500 | 45 |
| UCMerced | 土地利用 | 2100 | 21 |
| LandCoverAI | 土地覆盖 | 10674 | 5 |
| BigEarthNet | 多标签分类 | 590k | 43 |

---

## 构建数据加载器

遥感任务经常需要自定义 `collate_fn`，因为数据集返回的是 dict 而非标准 `(X, y)` 元组：

```python
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda x: x.float() / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    images = torch.stack([transform(b['image']) for b in batch])
    labels = torch.tensor([b['label'] for b in batch])
    return images, labels

loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
```

> **提示**：遥感图像归一化建议沿用 ImageNet 的均值和标准差。EuroSAT 等数据集本身就是 RGB 三通道，与自然图像分布接近。

---

## 迁移学习：ResNet18 分类卫星图

直接用预训练 ResNet，只替换最后的全连接层：

```python
from torchvision.models import resnet18

model = resnet18(weights=None)  # 或 weights='IMAGENET1K_V1'
model.fc = nn.Linear(512, 10)   # 10 类输出
```

完整训练循环（3 个 epoch 就能达到不错的准确率）：

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**实际效果**（EuroSAT，RTX4060 GPU，batch=128，3 epoch 仅 40 秒）：

| Epoch | Loss | 训练准确率 |
|-------|------|-----------|
| 1 | 0.36 | 94.7% |
| 2 | 0.18 | 96.8% |
| 3 | 0.15 | 97.8% |

测试集准确率 **83.7%**。CPU 上一个 epoch 要十多分钟，GPU 三个 epoch 只用了 40 秒——效果提升显著。

---

## 完整代码

上面的完整可运行代码见 `demo.py`，关键步骤总结：

```
加载数据 → 划分训练/测试 → 预处理 → DataLoader
    ↓
ResNet18 + 替换分类头 → 训练 3 epoch
    ↓
测试评估 → 可视化预测结果
```

运行：

```bash
pip install torchgeo torch torchvision matplotlib
python demo.py
```

---

## 接下来学什么？

TorchGeo 的能力远不止分类。掌握上面的基础后，可以深入：

1. **地理空间采样器** — `RandomGeoSampler` / `GridGeoSampler`，从大尺寸遥感图中按地理坐标采样小块
2. **预训练遥感模型** — TorchGeo 内置了在 BigEarthNet 上预训练的 ResNet/FCN 权重
3. **语义分割** — 用 `LandCoverAI` + `DeepLabV3` 做像素级地物分类
4. **多光谱数据处理** — 处理 Sentinel-2 的 13 波段图像
5. **变化检测** — 对比不同时相的卫星图像，检测地表变化

官方文档：[docs.torchgeo.org](https://docs.torchgeo.org)

---

## 参考

- [TorchGeo GitHub](https://github.com/microsoft/torchgeo)
- [EuroSAT 论文](https://arxiv.org/abs/1709.00029)
- [TorchGeo 官方教程](https://docs.torchgeo.org/en/latest/tutorials/)

---

*首次发布于 2026-05-20 · 掘金 / 知乎*
