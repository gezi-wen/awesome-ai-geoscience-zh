# GISclaw：让 LLM 替你操作 GIS

> 纯 Python 开源 LLM Agent，在 GeoAnalystBench 上达到 100% 成功率。不用 ArcGIS，不用 QGIS——只需要自然语言。

---

## 一句话概括

**GISclaw** 是一个 LLM 驱动的全栈地理空间分析 Agent。你对它说「计算这个区域的洪水风险」，它自动写 Python 代码、调用开源 GIS 库、生成分析结果和地图。

特色：**零商业 GIS 依赖**。不用 ArcGIS、不用 ENVI、不用 QGIS 插件——纯 Python 开源栈完成一切。

---

## 为什么值得关注？

### 1. 打破商业 GIS 壁垒

传统 GIS 教学和科研严重依赖商业软件（ArcGIS 年费数万元）。GISclaw 证明：GeoPandas + rasterio + scipy + cartopy 这一套开源组合，加上 LLM 的推理能力，可以完成同等甚至更复杂的分析。

### 2. 100% 基准成功率

在 **GeoAnalystBench**（50 个多步骤 GIS 任务，平均 5.8 步/任务）上，双 Agent 模式达到 100% 成功率：

| 模式 | 成功率 | 平均步数 |
|------|--------|---------|
| 单 Agent (ReAct) | ~85% | 7.2 |
| 双 Agent (Plan-Execute-Replan) | **100%** | 6.1 |

任务覆盖：空间连接、栅格代数、克里金插值、ML 分类、网络分析、多图层专题制图。

---

## 两种 Agent 架构

### 单 Agent (ReAct)

```
用户任务 → Think → Act → Observe → Think → Act → ... → 完成
```

最简单的形式。Agent 在思维和行动间循环，边想边做。

### 双 Agent (Plan-Execute-Replan)

```
用户任务 → Planner (制定计划) → Worker (执行)
                                     ↓
                                  出错?
                                     ↓
                                  Replan (重新规划)
```

更稳健。Planner 先把任务拆成子步骤，Worker 逐一执行。遇到错误时触发 Replan——这是达到 100% 成功率的关键。

---

## 三大工程化创新

### Schema Analysis（模式分析）

Agent 在写代码前，先读取数据的 schema（字段名、数据类型、CRS、分辨率）。避免写出 `df['population']` 但实际列名是 `df['pop_2020']` 这种常见 LLM 错误。

### Package Constraint（包约束）

严格按照「只能用开源 Python 包」的原则生成代码。如果 LLM 想调用 `arcpy.Buffer_analysis()`，包约束规则会在 prompt 里阻止它，告诉它用 `geopandas.GeoDataFrame.buffer()`。

### Error Memory（错误记忆）

跨任务的错误记忆模块。一次任务中踩过的坑（比如某个 shapefile 的编码问题），会被记录下来，后续任务自动避开。

---

## 代码结构

```
GISclaw/src/agent/
├── react_agent.py   → ReAct 循环主逻辑
├── sandbox.py       → 持久化 Jupyter-like 沙箱
├── tools.py         → Agent 工具集
├── llm_engine.py    → LLM 后端（云端 API / 本地模型）
├── error_memory.py  → 跨任务错误→修复记忆
├── prompts.py       → 系统提示构建器
├── orchestrator.py  → 双 Agent 协调器
├── planner.py       → Planning Agent
└── worker.py        → Execution Worker

GISclaw/src/tools/
├── vector_tools.py    → 矢量操作
├── raster_tools.py    → 栅格操作
├── analysis_tools.py  → 空间分析
├── viz_tools.py       → 可视化
├── terrain_tools.py   → 地形分析
└── advanced_tools.py  → ML 分类 / 变化检测
```

---

## 支持的 LLM 后端

- **云端 API**：GPT-4o、GPT-4.1、DeepSeek-V3、Claude（通过 litellm 统一接口）
- **本地模型**：Llama-3-70B、Qwen-2.5-72B、DeepSeek-R1 等开源模型

本地部署建议用 14B+ 参数的模型，更小的模型在复杂 GIS 推理上容易出错。

---

## 与技术栈中其他项目的关系

```
TorchGeo (数据加载) → Prithvi/DeepEarth (特征提取)
                              ↓
                        GISclaw (空间分析)
                              ↓
                     EVE (文献检索) → SciAssistant (报告生成)
```

GISclaw 填补了「数据→特征」之后的空间分析缺口。它和 TorchGeo 可以联合：TorchGeo 加载数据 → GISclaw 分析。

---

## 局限

1. **项目很新**：GitHub 只有 2 颗星（2026 年 3 月的论文），文档和社区几乎为零
2. **LLM 成本**：复杂任务每次调用可能消耗数十万 tokens
3. **结果可复现性**：LLM 输出有随机性，同一任务两次执行可能得到不同代码
4. **本地模型要求**：14B+ 参数模型需要 A100/H100 级别的 GPU

---

## 参考

- [GISclaw GitHub](https://github.com/geumjin99/GISclaw)
- [论文 arXiv:2603.26845](https://arxiv.org/abs/2603.26845)
- [GeoAnalystBench](https://github.com/geumjin99/GeoAnalystBench)

---

*首次发布于 2026-05-20 · 掘金 / 知乎*
