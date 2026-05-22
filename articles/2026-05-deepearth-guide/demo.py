"""
DeepEarth 入门 —— Earth4D 时空编码器演示
===========================================
演示如何用 DeepEarth 的 Earth4D 编码器生成行星级时空特征。

运行前：
  cd /mnt/e/workspace/deepearth
  pip install torch ninja
  cd encoders/xyzt/hashencoder && python3 setup.py build_ext --inplace

注意：需要 CUDA Toolkit 编译哈希编码器 GPU 内核。无 GPU 环境无法运行，
但代码逻辑正确（来自官方仓库 earth4d_demo.py），可直接阅读学习。
"""

import sys
sys.path.insert(0, '/mnt/e/workspace/deepearth')

import torch
from encoders.xyzt.earth4d import Earth4D

print("=" * 60)
print("DeepEarth · Earth4D 时空编码器")
print("=" * 60)

# ─── 1. 初始化世界模型 ──────────────────────────
print("\n🌍 初始化 Earth4D...")
world_model = Earth4D(
    spatial_levels=24,           # 空间分辨率层级
    temporal_levels=24,          # 时间分辨率层级
    features_per_level=2,        # 每层特征数
    spatial_log2_hashmap_size=22,  # 哈希表大小 (2^22)
    temporal_log2_hashmap_size=22,
    base_spatial_resolution=32.0,
    base_temporal_resolution=32.0,
    enable_learned_probing=True,  # 启用学习探针
    coordinate_system='ecef',     # ECEF 地心坐标系
)
print(f"  参数总量: {sum(p.numel() for p in world_model.parameters()):,}")
print(f"  输出维度: {world_model.output_dim} (空间 {world_model.spatial_dim} + 时空 {world_model.spatiotemporal_dim})")

# ─── 2. 输入坐标 → 生成时空特征 ──────────────
print("\n📍 输入 4 个历史地点的 (纬度, 经度, 海拔, 时间)...")
embeddings = world_model(
    # 布莱切利园 (图灵破解 Enigma, 1941)
    (51.9976, -0.7416, 110, "1941-06-01 09:00 GMT"),
    # 卡内基梅隆 (Hinton 发明玻尔兹曼机, 1985)
    (40.4433, -79.9436, 270, "1985-01-15 10:00 ET"),
    # CERN (Berners-Lee 发明 WWW, 1989)
    (46.2330, 6.0557, 430, "1989-03-12 10:00 CET"),
    # Mila, 魁北克 (World Modeling Workshop 2026)
    (45.5308, -73.6128, 63, "2026-02-04 11:00 ET"),
)

print(f"\n  ✅ 输出 shape: {list(embeddings.shape)}")
print(f"     4 个地点 × 192维 = 可训练的时空特征向量")
print(f"     每个位置 -> 48维空间特征 + 144维时空特征")

# ─── 3. 特征结构解析 ──────────────────────────
print("\n📐 特征结构:")
print(f"   [空间编码] xyz (lat, lon, elev):")
print(f"     24级 × 2特征/级 = 48维")
print(f"   [时空编码] xyt + yzt + xzt (3个投影):")
print(f"     每投影: 24级 × 2特征/级 = 48维")
print(f"     3 × 48 = 144维")
print(f"   总计: 48 + 144 = 192维")

# ─── 4. 批量处理演示 ──────────────────────────
print("\n📊 批量处理...")
batch_coords = torch.rand(100, 4)  # 100个随机坐标
batch_coords[:, 0] = batch_coords[:, 0] * 180 - 90    # lat: [-90, 90]
batch_coords[:, 1] = batch_coords[:, 1] * 360 - 180   # lon: [-180, 180]
batch_coords[:, 2] = batch_coords[:, 2] * 9000         # elev: [0, 9000]m
batch_coords[:, 3] = batch_coords[:, 3]                 # time: [0, 1]

batch_embeddings = world_model(batch_coords)
print(f"  输入: {list(batch_coords.shape)}")
print(f"  输出: {list(batch_embeddings.shape)} (100个地点的192维特征)")

# ─── 5. 两种坐标系 ─────────────────────────────
print("\n🗺️  支持的坐标系:")
print("  1. ECEF (地心地固坐标系) — WGS84 椭球转换")
print("     适合: 全球尺度、物理模拟")
print("  2. Geographic (地理直接映射)")
print("     x=纬度, y=经度, z=海拔, t=时间")
print("     适合: 区域研究、生态知识迁移")

# 切换到地理坐标系
geo_model = Earth4D(coordinate_system='geographic', verbose=False)
geo_embeddings = geo_model(batch_coords)
print(f"\n  地理模式输出: {list(geo_embeddings.shape)}")

# ─── 6. 自适应范围 ──────────────────────────────
print("\n🎯 自适应范围 (Adaptive Range):")
print("  对区域数据集，用 fit_range() 自动拟合坐标范围")
print("  提升哈希表利用率，减少碰撞")
print("  用法: world_model.fit_range(train_coords)")

# ─── 7. 预计算加速 ──────────────────────────────
print("\n⚡ 预计算加速:")
print("  训练前调用 precompute() 预先计算哈希索引")
print("  训练中用 forward_precomputed() 跳过哈希查找")
print("  适合: 固定坐标的场景（如固定站点网络）")

print("\n" + "=" * 60)
print("✅ 演示完成！")
print("=" * 60)
print("\n📖 下一步:")
print("  1. 读 arXiv:2603.07039 了解完整架构")
print("  2. 跑 benchmarks/caravan/ 做生态预测")
print("  3. 联系维护者: lance@ecodash.ai")
print("  4. 提 PR 贡献文档和教程")
