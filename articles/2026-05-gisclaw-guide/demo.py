"""
GISclaw 演示 —— LLM 驱动的开源地理空间分析 Agent
===================================================
GISclaw 是一个纯 Python 开源 LLM Agent，不依赖任何商业 GIS 软件，
可以完成空间叠加、栅格代数、克里金插值、ML 分类等专业 GIS 任务。

论文: https://arxiv.org/abs/2603.26845
仓库: https://github.com/geumjin99/GISclaw

运行前: pip install geopandas rasterio scikit-learn scipy libpysal cartopy
"""

print("=" * 60)
print("GISclaw · LLM 地理空间分析 Agent")
print("=" * 60)

# ─── 1. 系统概览 ──────────────────────────────
print("""
┌─────────────────────────────────────────────────────────┐
│                     GISclaw 架构                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户输入: 「计算这个区域的城市热岛指数」                  │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │  LLM Agent (ReAct / Plan-Execute-Replan)         │   │
│  │  • Schema Analysis: 理解地理数据格式               │   │
│  │  • Package Constraint: 严格使用开源 Python 栈      │   │
│  │  • Domain Knowledge: 可注入领域知识                │   │
│  │  • Error Memory: 跨任务错误→修复记忆               │   │
│  └──────────────────────────────────────────────────┘   │
│       ↓                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Python Sandbox (持久化 Jupyter-like)             │   │
│  │  • GeoPandas → 矢量分析                           │   │
│  │  • rasterio  → 栅格代数                           │   │
│  │  • scipy     → 克里金插值                          │   │
│  │  • sklearn   → ML 分类                            │   │
│  │  • cartopy   → 专题制图                            │   │
│  └──────────────────────────────────────────────────┘   │
│       ↓                                                 │
│  输出: 分析结果 + 地图 + 报告                            │
└─────────────────────────────────────────────────────────┘
""")

# ─── 2. 两种 Agent 架构 ─────────────────────────
print("=" * 60)
print("两种 Agent 架构")
print("=" * 60)
print("""
单 Agent (ReAct):
  思考 → 行动 → 观察 → 思考 → ...
  适合: 简单到中等复杂度的任务
  优点: 简单、快速

双 Agent (Plan-Execute-Replan):
  Planner: 分析任务 → 制定执行计划 → 分解子步骤
  Worker:  执行子步骤 → 遇到错误 → 触发 Replan
  适合: 复杂的多步骤 GIS 流程
  优点: 100% GeoAnalystBench 成功率
""")

# ─── 3. 核心模块 ────────────────────────────────
print("=" * 60)
print("核心模块")
print("=" * 60)
print("""
GISclaw/src/agent/

├── react_agent.py    → ReAct 循环主逻辑
├── sandbox.py        → 持久化 Python 沙箱
├── tools.py          → 暴露给 Agent 的工具集
├── llm_engine.py     → LLM 后端 (云端API / 本地14-70B)
├── error_memory.py   → 错误记忆 (跨任务学习)
├── prompts.py        → 系统提示构建器
├── orchestrator.py   → 双Agent 协调器
├── planner.py        → 任务规划者
└── worker.py         → 任务执行者

GISclaw/src/tools/

├── registry.py       → 工具注册表
├── vector_tools.py   → 矢量操作 (buffer, overlay, join)
├── raster_tools.py   → 栅格操作 (algebra, resample, zonal)
├── analysis_tools.py → 空间分析 (kriging, hotspot, network)
├── viz_tools.py      → 可视化 (choropleth, multi-layer map)
├── conversion_tools.py → 格式转换 (shp↔geojson, crs转换)
├── terrain_tools.py  → 地形分析 (slope, aspect, viewshed)
└── advanced_tools.py → 高级功能 (ML分类, 变化检测)
""")

# ─── 4. 不用商业 GIS 的秘诀 ─────────────────────
print("=" * 60)
print("纯开源技术栈 (零商业依赖)")
print("=" * 60)
print("""
传统 GIS                        GISclaw
────────                        ───────
ArcGIS (商业)        →          GeoPandas + shapely
ENVI (商业)          →          rasterio + xarray
ArcPy (商业)         →          scipy + numpy
QGIS (桌面)          →          cartopy + matplotlib
ArcGIS Server        →          FastAPI (可选部署)

三大工程化创新:
1. Schema Analysis: Agent 先读数据schema再写代码
2. Package Constraint: 严格限制只用开源 Python 包
3. Error Memory: 错误被记住→下次类似任务自动避开
""")

# ─── 5. 使用示例 ────────────────────────────────
print("=" * 60)
print("使用示例 (概念)")
print("=" * 60)
print("""
from src.agent.react_agent import ReActAgent

# 初始化 Agent
agent = ReActAgent(
    llm_backend="openai",   # 或 "deepseek", "local"
    sandbox_persist=True,    # 跨任务保持变量
)

# 任务 1: 空间叠加
result = agent.run(\"""
    加载 parks.shp 和 flood_zones.shp，
    计算洪水区域内公园面积占比，
    输出 choropleth 地图。
\""")

# 任务 2: 地形分析
result = agent.run(\"""
    从 dem.tif 计算坡度和坡向，
    提取坡度 > 15° 的区域，
    与土地利用数据叠加。
\""")

# 任务 3: ML 分类
result = agent.run(\"""
    用 Sentinel-2 波段做随机森林土地覆盖分类，
    输出分类图 + 混淆矩阵 + 精度报告。
\""")

print(f"任务完成: {result['status']}")
print(f"使用工具: {result['tools_used']}")
""")

# ─── 6. 性能 ───────────────────────────────────
print("=" * 60)
print("GeoAnalystBench 基准成绩")
print("=" * 60)
print("""
数据集: 50 个多步骤 GIS 任务，平均 5.8 步/任务

单 Agent (ReAct + GPT-4o):
  成功率: ~85%
  平均步数: 7.2

双 Agent (Plan-Execute-Replan + GPT-4o):
  成功率: 100% ★
  平均步数: 6.1

  任务类型覆盖:
  ✅ 空间连接 (Spatial Join)
  ✅ 栅格代数 (Raster Algebra)
  ✅ 克里金插值 (Kriging)
  ✅ ML 分类 (Random Forest / SVM)
  ✅ 网络分析 (Network Analysis)
  ✅ 专题制图 (Multi-layer Cartography)
""")

print("=" * 60)
print("✅ GISclaw 架构演示完成")
print("=" * 60)
print("""
📖 关键收获:
  1. LLM Agent 可以做专业的全栈 GIS 分析
  2. 不依赖任何商业软件，纯 Python 开源栈
  3. Error Memory 让 Agent 越用越好
  4. 支持云端 API (GPT/DeepSeek) 和本地开源模型

🔗 论文: https://arxiv.org/abs/2603.26845
📦 仓库: https://github.com/geumjin99/GISclaw
""")
