# 从卫星图像到科研报告：AI + 地球科学技术栈全解析

> 用四个开源项目串起一条完整的 AI 地学链路：数据加载 → 时空编码 → 知识检索 → 自动写作。

---

## 开篇

遥感 + 深度学习的交叉领域正在爆发。但初学者面对海量工具和论文时，最大的困惑是——**这些东西之间是什么关系？应该从哪个开始？**

这篇文章用四个开源项目，串起一条从「拿到卫星图」到「写出研究报告」的完整链路。

---

## 四层技术栈

```
┌────────────────────────────────────────────┐
│  Layer 4: 自动写作                          │
│  SciAssistant (中科院南海所 × 华为盘古)      │
│  多智能体协作：Planner → Seeker → Writer    │
├────────────────────────────────────────────┤
│  Layer 3: 知识检索                          │
│  EVE (ESA Φ-lab × Mistral AI)             │
│  RAG + 24B 地球科学 LLM + MCP 协议          │
├────────────────────────────────────────────┤
│  Layer 2: 时空编码                          │
│  DeepEarth (Allen AI × Stanford × ASU)     │
│  Earth4D: 4D 哈希编码 → 192维时空特征        │
├────────────────────────────────────────────┤
│  Layer 1: 数据加载                          │
│  TorchGeo (PyTorch 官方)                    │
│  50+ 遥感数据集 + 地理感知采样器              │
└────────────────────────────────────────────┘
```

### Layer 1：TorchGeo — 遥感数据的 PyTorch

**一句话**：让你用 PyTorch 处理遥感图像像处理普通图片一样简单。

遥感数据有三个天然痛点：地理坐标系（GeoTIFF 而不是 PNG）、多光谱通道（不止 RGB）、超大尺寸（不能直接塞 GPU）。TorchGeo 解决了这三个问题：

```python
from torchgeo.datasets import EuroSAT

dataset = EuroSAT(root="./data", download=True)
# 27000 张 Sentinel-2 卫星图像，10 类土地覆盖
# 返回 dict: {'image': tensor, 'label': 0..9}

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
# 直接输入 PyTorch 训练循环
```

**关键能力**：50+ 内置数据集，一行下载；地理空间采样器处理大图切分；与 torchvision 无缝对接。

**适合谁**：刚入门遥感 DL 的学生。先跑通一个 EuroSAT 分类，理解数据格式，再往上走。

---

### Layer 2：DeepEarth — 给地球一个时空大脑

**一句话**：输入地球上任意一个位置 + 时间，输出 192 维的「环境状态」特征向量。

传统深度学习把卫星图当像素矩阵处理，丢失了最关键的信息——**这是地球**。同一片森林在春天和秋天看起来截然不同，但它们本质上是同一个生态系统。

DeepEarth 把「空间 + 时间」的知识编码进模型结构：

```python
from deepearth.encoders.xyzt.earth4d import Earth4D

world_model = Earth4D()
embeddings = world_model(
    (51.9976, -0.7416, 110, "1941-06-01 09:00 GMT"),
    (45.5308, -73.6128, 63, "2026-02-04 11:00 ET"),
)
# → [2, 192] 维时空特征向量
```

**核心创新**：
- **Earth4D 编码器**：把 4D 坐标（经度、纬度、海拔、时间）编码为 192 维特征
- **4 个 3D 投影**：xyz（纯空间）+ xyt、yzt、xzt（三个时空切片）
- **Learned Hash Probing**：用可学习的探针消解哈希碰撞 → 23% 误差降低 + 99% 参数缩减
- **NSF/DOE 超算支持**：获美国能源部 NERSC 超算资源

**在 LFMC（野火风险预测）基准上，5M 参数的 DeepEarth 超越了 500M+ 参数的通用模型。**

**适合谁**：有一定 ML 基础，想做地球科学前沿研究的人。读 arXiv:2603.07039，跑 Caravan 基准。

---

### Layer 3：EVE — 给 LLM 装上地球科学知识库

**一句话**：ESA 官方的地球科学 RAG 助手，用 Mistral 24B + 权威文献回答地学问题。

通用大模型（GPT、Claude）在地学问题上容易胡编。EVE 的解法是 RAG——回答前先检索 ESA 的地球观测文献库：

```
用户：「NDVI 为什么用近红外波段而不是红光？」
  ↓
检索 top-5 相关文献（EVE 知识库）
  ↓
拼接上下文 + Mistral Small 3.2 (24B)
  ↓
流式输出：引用论文支撑的回答
```

**三种接入方式**：
- **Python 客户端**：`pip install eve-api`，异步 JWT 自动刷新
- **MCP 协议**：在 Claude Code / Cursor 中直接调用 `query_eve`
- **直接 HTTP**：任何语言，REST API

**MCP 工具中特别有价值的两个**：`extract_factuality_issues` 和 `assess_factuality_issue`——不是简单问答，而是对 Google Earth Engine Python 脚本进行科学假设审查。

**适合谁**：需要在地学项目中集成知识检索的开发者。RAG + MCP 的双重标准让 EVE 可以嵌入任何 AI 工具链。

---

### Layer 4：SciAssistant — 让 AI 帮你写研究报告

**一句话**：中科院南海所的 Multi-Agent 系统，Planner 分解任务 → Seeker 搜索文献 → Writer 生成报告。

```
用户：「写一篇南海珊瑚礁遥感监测的研究综述」

Planner: 「这需要文献检索 + 长篇写作」
  ├── 子任务 1: 搜索珊瑚礁遥感文献 → Info Seeker
  ├── 子任务 2: 搜索南海区域研究 → Info Seeker
  └── 子任务 3: 撰写综述 → Writer

Info Seeker (ReAct 循环):
  思考 → 搜索 PubMed/ArXiv → 分析结果 → 调整策略 → 再搜

Writer (ReAct 循环):
  接收文献 → 发现缺高光谱部分 → 请求补充 → 生成完整综述

输出: Markdown + PDF 报告 (万字级)
```

**核心特征**：
- 三智能体协作（Planner / Seeker / Writer），各遵循 ReAct 模式
- 基于华为盘古 DeepDiver-V2，通过 litellm 支持替换为 DeepSeek 等兼容 API
- DeepDiver 模式最多 40 轮迭代，适合生成完整研究报告
- MCP 协议接入 PubMed / ArXiv / Google Search

**适合谁**：需要自动化文献综述和研究报告生成的科研人员。

---

## 完整链路实战

以一个具体任务串联四层：

> **目标**：分析 2023-2025 年加州森林火灾风险的时空变化，生成研究报告。

| 步骤 | 工具 | 做什么 |
|------|------|--------|
| 1 | **TorchGeo** | 加载 Sentinel-2 卫星图像，按地理坐标采样加州区域 |
| 2 | **DeepEarth** | 对每个采样点生成时空特征向量，编码火灾风险模式 |
| 3 | **EVE** | 检索 ESA 文献库中关于加州野火的最新研究 |
| 4 | **SciAssistant** | 整合特征分析结果 + 文献综述 → 自动生成报告 |

**四层不是孤立的**。TorchGeo 产出的图像可以直接输入 DeepEarth 的时空编码器；DeepEarth 的预测结果可以作为 EVE RAG 查询的上下文；EVE 检索的文献和 DeepEarth 的分析结果一起，成为 SciAssistant 写报告的素材。

---

## 选型指南

| 角色 | 从哪里开始 | 理由 |
|------|-----------|------|
| 本科生/入门 | TorchGeo | 门槛最低，跟 PyTorch 无缝，学了就能用 |
| 研究生/方向确定 | DeepEarth | 时空编码是地学 AI 的前沿，论文级产出 |
| 应用开发者 | EVE | API 最友好，MCP 协议可嵌入所有工具 |
| 科研人员 | SciAssistant | 自动化文献综述，节省大量时间 |

如果只选一个开始——**TorchGeo**。跑通遥感分类的全流程只需要一个下午。

如果要追求前沿和差异化——**DeepEarth**。时空编码 + 地球先验是目前地学 AI 最活跃的方向。

---

## 四个项目速查

| 项目 | 仓库 | 论文 | 背景 |
|------|------|------|------|
| TorchGeo | GitHub microsoft/torchgeo | - | PyTorch 官方 |
| DeepEarth | GitHub legel/deepearth | arXiv:2603.07039 | Allen AI × Stanford |
| EVE | GitHub FrontierDevelopmentLab | - | ESA Φ-lab |
| SciAssistant | GitHub scsio-marinebio | - | 中科院南海所 |

---

*首次发布于 2026-05-20 · 掘金 / 知乎*
