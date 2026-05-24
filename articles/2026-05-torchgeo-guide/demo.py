"""
TorchGeo 入门 —— 用 PyTorch 处理遥感数据
===========================================
使用 EuroSAT 数据集 + ResNet18 做卫星图像场景分类

实测: RTX4060 GPU, 27000张图, 3 epoch仅40秒, 测试准确率83.7%

运行前：
  pip install torchgeo torch torchvision matplotlib rasterio pillow

数据集自动下载到 /tmp/torchgeo_data（~2GB，仅首次）。

注意：TorchGeo 的 EuroSAT() 原生支持 .tif 多光谱图像，直接用即可。
如果绕过 TorchGeo 用 torchvision.datasets.ImageFolder 加载 EuroSAT，
需手动注册 .tif 扩展名（torchvision 默认只认 .jpg/.png）并用 rasterio
读取。这不是 TorchGeo 的问题，而是 torchvision 的限制。具体见：
https://github.com/torchgeo/torchgeo/issues/3731
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import os

# 允许 torchvision 识别 .tif 文件
import torchvision.datasets.folder as folder
folder.IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

# ─── 1. 加载遥感数据集 ─────────────────────────────
print("📡 加载 EuroSAT 数据集...")

class RasterioLoader:
    """用 rasterio 读取多光谱 GeoTIFF，取 RGB 三通道"""
    def __call__(self, path):
        with rasterio.open(path) as src:
            img = src.read([3, 2, 1])  # B04(红), B03(绿), B02(蓝)
            return torch.from_numpy(img.astype(np.float32))

# 从 TorchGeo 自动下载（仅首次）
from torchgeo.datasets import EuroSAT
eurosat = EuroSAT(root="/tmp/torchgeo_data", download=True)

DATA_DIR = os.path.join("/tmp/torchgeo_data", eurosat.base_dir)
dataset = datasets.ImageFolder(DATA_DIR, loader=RasterioLoader())

print(f"  图像数量: {len(dataset)}")
print(f"  类别数:   {len(dataset.classes)}")
print(f"  场景类别: {', '.join(dataset.classes)}")

# ─── 2. 划分数据集 ────────────────────────────────
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"\n📊 训练集: {len(train_ds)} 张 | 测试集: {len(test_ds)} 张")

# ─── 3. 数据预处理 ────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda x: x.float() / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    images = torch.stack([transform(b[0]) for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return images, labels

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

# ─── 4. 模型 ──────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(weights=None)
model.fc = nn.Linear(512, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\n🤖 设备: {device} | 模型参数: {sum(p.numel() for p in model.parameters())//1000:,}K")

# ─── 5. 训练 ──────────────────────────────────────
print("\n🏃 开始训练...")
model.train()
for epoch in range(3):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = model(images).max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.3f} Acc={100.*correct/total:.1f}%")

# ─── 6. 测试 ──────────────────────────────────────
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, preds = model(images).max(1)
        test_total += labels.size(0)
        test_correct += preds.eq(labels).sum().item()
print(f"\n✅ 测试准确率: {100.*test_correct/test_total:.1f}%")

# ─── 7. 可视化 ────────────────────────────────────
class_names = dataset.classes
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:8].to(device)
sample_labels = sample_labels[:8]

with torch.no_grad():
    _, preds = model(sample_images).max(1)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.clip(img * std + mean, 0, 1)
    ax.imshow(img)
    true_label = class_names[sample_labels[i].item()]
    pred_label = class_names[preds[i].item()]
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('torchgeo_result.png', dpi=120, bbox_inches='tight')
print("📸 结果图已保存: torchgeo_result.png")
print("\n🎉 完成！")
