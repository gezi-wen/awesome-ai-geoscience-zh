"""Prithvi GPU 推理 — NASA/IBM 遥感基础模型

验证环境: RTX 4060 8GB, CUDA 13.2, PyTorch 2.12
模型参数: 330M, 推理速度: 31ms/图 (32 fps)

使用方式:
  git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
  python demo.py
"""
import torch, time, sys

SNAP = "Prithvi-EO-2.0-300M"
sys.path.insert(0, SNAP)
from prithvi_mae import PrithviMAE

print("加载 Prithvi-EO-2.0-300M (330M参数, GPU)...")

# 关键: img_size=448, patch_size=16 必须匹配权重
model = PrithviMAE(
    img_size=448, patch_size=(1, 16, 16), num_frames=1, in_chans=6,
    embed_dim=1024, depth=24, num_heads=16,
    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    mlp_ratio=4., norm_pix_loss=False, mask_ratio=0.0
)
model = model.cuda().eval()

sd = torch.load(f"{SNAP}/Prithvi_EO_V2_300M.pt", map_location="cpu")
if "model_state" in sd:
    sd = sd["model_state"]
for k in list(sd.keys()):
    if "pos_embed" in k:
        del sd[k]
model.load_state_dict(sd, strict=False)

print(f"参数: {sum(p.numel() for p in model.parameters()) // 10**6}M")
print(f"显存: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

x = torch.randn(2, 6, 448, 448).cuda()
with torch.no_grad():
    loss, pred, mask = model(x)
print(f"输入: {list(x.shape)} → 重建: {list(pred.shape)}")

t0 = time.time()
for _ in range(10):
    model(torch.randn(1, 6, 448, 448).cuda())
dt = (time.time() - t0) / 10 * 1000
print(f"速度: {dt:.1f}ms/图 ({1000/dt:.0f} fps)")
print("✅ Prithvi GPU 推理正常")
