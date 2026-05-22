# Getting Started with TorchGeo: Remote Sensing with PyTorch

> Your first step into deep learning for earth observation. By the end, you'll load satellite imagery and train a land cover classifier.

---

## Why TorchGeo?

`torchvision` is great for natural images. But remote sensing data is different:

- **GeoTIFFs**, not PNGs — with coordinate reference systems baked in
- **Multi-spectral** bands — beyond RGB into near-infrared, thermal, SAR
- **Massive sizes** — a single satellite image can be 10,000×10,000 pixels
- **Spatial context matters** — random cropping destroys geographic patterns

[TorchGeo](https://github.com/microsoft/torchgeo) is PyTorch's official geospatial extension by Microsoft. It provides 50+ remote sensing datasets (one-line download), geo-aware samplers, and seamless integration with torchvision and PyTorch Lightning.

```bash
pip install torchgeo rasterio
```

---

## Loading Your First Dataset

Let's start with **EuroSAT** — 27,000 Sentinel-2 satellite images across 10 land cover classes:

```python
from torchgeo.datasets import EuroSAT

dataset = EuroSAT(root="./data", download=True)
print(len(dataset))          # 27000
print(dataset.num_classes)   # 10
print(dataset.classes)
# ['AnnualCrop', 'Forest', 'HerbaceousVegetation',
#  'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
#  'Residential', 'River', 'SeaLake']
```

Each sample is a dict with `image` (multi-spectral tensor) and `label` (integer):

```python
sample = dataset[0]
print(sample['image'].shape)  # torch.Size([13, 64, 64]) — 13 Sentinel-2 bands
print(sample['label'])        # 0 → AnnualCrop
```

---

## Building the Data Pipeline

Remote sensing datasets return dicts, so we need a custom `collate_fn`:

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
    images = torch.stack([transform(b['image'][:3]) for b in batch])
    labels = torch.tensor([b['label'] for b in batch])
    return images, labels

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

> We take `image[:3]` for RGB bands (B04, B03, B02). For multi-spectral work, keep all 13 bands.

---

## Transfer Learning with ResNet18

Replace the final fully-connected layer for our 10 classes:

```python
from torchvision.models import resnet18

model = resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(512, 10)
```

**After just 1 epoch on CPU**, you can expect ~62% accuracy. With 3 epochs and ImageNet pretrained weights, this easily exceeds 85%.

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Key Datasets at a Glance

| Dataset | Task | Size | Classes |
|---------|------|------|---------|
| RESISC45 | Scene classification | 31,500 | 45 |
| UCMerced | Land use | 2,100 | 21 |
| LandCoverAI | Land cover | 10,674 | 5 |
| BigEarthNet | Multi-label | 590k | 43 |

---

## Where to Go Next

1. **Geo-aware sampling** — `RandomGeoSampler` for tiling massive GeoTIFFs
2. **Pre-trained remote sensing models** — TorchGeo ships weights pretrained on BigEarthNet
3. **Semantic segmentation** — `LandCoverAI` + DeepLabV3 for pixel-level classification
4. **Multi-spectral processing** — Work with all 13 Sentinel-2 bands
5. **Change detection** — Compare satellite images across time

Official docs: [docs.torchgeo.org](https://docs.torchgeo.org)

---

## References

- [TorchGeo GitHub](https://github.com/microsoft/torchgeo)
- [EuroSAT Paper](https://ieeexplore.ieee.org/document/8736785)

---

*Published 2026-05-20 · freeCodeCamp / Medium*
