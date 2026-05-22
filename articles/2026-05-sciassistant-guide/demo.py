"""
SciAssistant 架构演示 —— 中科院南海所多智能体科研助手
========================================================
演示 SciAssistant 的核心架构：Planner → Information Seeker → Writer 三智能体协作。

本项目基于华为盘古 DeepDiver-V2 大模型，采用多智能体协作架构（MAS）。
运行需要 MySQL、Pangu API 端点和深度学习环境。

本 demo 展示架构概念和关键代码路径，不依赖外部服务。
"""

print("=" * 60)
print("SciAssistant · 多智能体科研助手架构")
print("=" * 60)

# ─── 1. 系统概览 ──────────────────────────────
print("""
┌─────────────────────────────────────────────────────────┐
│                    SciAssistant                          │
│  中科院南海海洋研究所 × 华为盘古 DeepDiver-V2            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │ Planner  │───▶│ Information      │───▶│  Writer  │  │
│  │ Agent    │    │ Seeker Agent     │    │  Agent   │  │
│  │ 任务规划  │    │ 信息检索          │    │ 报告生成  │  │
│  └──────────┘    └──────────────────┘    └──────────┘  │
│       │                  │                     │       │
│       ▼                  ▼                     ▼       │
│  ┌─────────────────────────────────────────────────┐   │
│  │         MCP Server (知识库 / 搜索引擎)           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  前端: Flask + REST API + WebSocket                     │
│  存储: MySQL (会话 / 用户 / 文件)                        │
│  模型: Pangu LLM (via litellm)                          │
└─────────────────────────────────────────────────────────┘
""")

# ─── 2. 三智能体协作流程 ─────────────────────
print("=" * 60)
print("三智能体协作流程")
print("=" * 60)

print("""
用户输入：「写一篇关于南海珊瑚礁遥感监测的研究综述」

Step 1 — Planner Agent (规划)
  ├─ 分析任务: 「这是一个需要深度信息检索 + 长篇写作的研究综述」
  ├─ 分解子任务:
  │   1. 搜索珊瑚礁遥感监测的最新文献
  │   2. 搜索南海区域的相关研究
  │   3. 搜索遥感方法（光学/SAR/高光谱）
  │   4. 综合以上信息撰写综述
  └─ 分配: 子任务 1-3 → Information Seeker, 子任务 4 → Writer

Step 2 — Information Seeker Agent (信息检索)
  ├─ ReAct 循环: Reasoning ↔ Acting
  │   ├─ 思考: 「需要从多个来源搜索相关论文」
  │   ├─ 行动: 调用 MCP 工具 → PubMed API / ArXiv API / Google Search
  │   ├─ 观察: 获取检索结果
  │   └─ 迭代: 结果不够 → 调整搜索策略 → 再搜
  └─ 返回: 结构化的文献检索结果

Step 3 — Writer Agent (报告撰写)
  ├─ 接收: Information Seeker 的结构化结果
  ├─ ReAct 循环:
  │   ├─ 思考: 「这些文献覆盖了光学遥感和 SAR，但缺少高光谱部分」
  │   ├─ 行动: 请求补充检索 → Information Seeker 再次搜索
  │   ├─ 观察: 补充文献到达
  │   └─ 写作: 生成完整综述
  └─ 输出: Markdown 格式的研究报告 (保存到 report/)

Step 4 — 返回用户
  └─ WebSocket 推送完成通知 → 用户下载报告
""")

# ─── 3. 三种工作模式 ─────────────────────────
print("=" * 60)
print("三种工作模式")
print("=" * 60)

print("""
┌──────────┬──────────────────┬──────────────────────┐
│ 模式     │ 说明             │ 适用场景             │
├──────────┼──────────────────┼──────────────────────┤
│ Chat     │ 标准 LLM 对话    │ 日常问答、简单咨询    │
│ Reasoner │ 深度推理 (CoT)   │ 复杂分析、需要思维链  │
│ DeepDiver│ 万字长文生成     │ 综述、报告、论文      │
└──────────┴──────────────────┴──────────────────────┘

DeepDiver 模式的独特之处：
  • max_iteration 可达 40 轮
  • 支持多轮信息检索 + 写作
  • 自动生成参考文献列表
  • 输出 Markdown + PDF 两种格式
""")

# ─── 4. 核心代码路径 ─────────────────────────
print("=" * 60)
print("核心代码路径（deepdiver_v2/）")
print("=" * 60)

print("""
deepdiver_v2/
├── src/
│   ├── agents/
│   │   ├── base_agent.py              → BaseAgent (ReAct 模式基类)
│   │   ├── planner_agent.py           → PlannerAgent (任务分解)
│   │   ├── objective_information_seeker.py → 客观信息检索
│   │   ├── subjective_information_seeker.py → 主观/观点检索
│   │   └── writer_agent.py            → WriterAgent (报告生成)
│   ├── tools/
│   │   ├── mcp_client.py              → MCP 客户端
│   │   ├── mcp_server_standard.py     → MCP 服务端标准
│   │   ├── paper.py                   → PDF 处理
│   │   └── normalizer.py              → 文本规范化
│   ├── workspace/
│   │   └── local_workspace_manager.py → 本地文件管理
│   └── utils/
│       └── task_manager.py            → 任务管理器
├── config/
│   ├── .env              → API 密钥、模型配置
│   └── config.py         → 配置定义（dataclass）
└── requirements.txt
""")

# ─── 5. LLM 配置示例 ─────────────────────────
print("=" * 60)
print("LLM 后端配置（支持 OpenAI 兼容 API）")
print("=" * 60)

print("""
# deepdiver_v2/config/.env
MODEL_REQUEST_URL=https://your-llm-api.com/v1/chat/completions
MODEL_REQUEST_TOKEN=your-api-key
MODEL_NAME=pangu_auto
MODEL_TEMPERATURE=0.6
MODEL_MAX_TOKENS=8192
MODEL_REQUEST_TIMEOUT=180

# 搜索配置（可选）
SEARCH_ENGINE_BASE_URL=https://your-search-api.com
SEARCH_ENGINE_API_KEYS=your-search-key

# 迭代次数
PLANNER_MAX_ITERATION=40
INFORMATION_SEEKER_MAX_ITERATION=30
WRITER_MAX_ITERATION=40
""")

print("=" * 60)
print("✅ 架构演示完成")
print("=" * 60)

print("""
📖 关键收获:
  1. SciAssistant 是标准的 Multi-Agent 协作架构
  2. 每个 Agent 遵循 ReAct 模式 (Reasoning ↔ Acting)
  3. Planner 分解任务 → Seeker 检索信息 → Writer 生成报告
  4. 支持替换任意 OpenAI 兼容 LLM 后端

🔗 仓库: https://github.com/scsio-marinebio/SciAssistant
📧 联系: 中科院南海海洋研究所
""")
