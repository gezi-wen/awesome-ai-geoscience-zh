# SciAssistant 架构解析：中科院南海所的多智能体科研助手

> 走进华为盘古 × 中科院南海所的 Multi-Agent 协作系统，理解科研 AI 的架构设计。

---

## 一句话概括

SciAssistant 是**中国科学院南海海洋研究所**基于华为盘古 DeepDiver-V2 大模型开发的多智能体科研助手。它能自动检索文献、分析文档、生成万字综述报告。

---

## 核心架构：三智能体协作

SciAssistant 采用经典的 Multi-Agent System (MAS) 架构，三个 Agent 各司其职、协同工作：

```
用户提问
    ↓
┌──────────────┐
│ Planner      │  分析任务 → 分解子问题 → 分配工作
└──────┬───────┘
       │ 子任务列表
       ↓
┌──────────────┐
│ Info Seeker  │  搜索文献 → 爬取网页 → 提取文档
└──────┬───────┘
       │ 结构化信息
       ↓
┌──────────────┐
│ Writer       │  综合分析 → 撰写报告 → 格式化输出
└──────┬───────┘
       ↓
  Markdown + PDF 报告
```

### Planner Agent（规划者）

- **职责**：接收用户指令，分析复杂度，分解为子任务
- **能力**：自动判断需要多少轮迭代（最多 40 轮）
- **模式切换**：auto（自动判断）/ writing（写作优先）/ qa（问答优先）

### Information Seeker Agent（信息搜寻者）

分为两个变体：

- **Objective Seeker**：客观事实检索。搜论文、查数据、爬网页
- **Subjective Seeker**：主观观点检索。找不同学派的立场、争议点

每个 Seeker 遵循 **ReAct 模式**（Reasoning + Acting）：
1. 思考：我需要什么信息？
2. 行动：调用 MCP 工具（PubMed / ArXiv / Google Search）
3. 观察：分析检索结果
4. 迭代：不够就调整策略再搜

### Writer Agent（写作者）

- **职责**：基于 Seeker 的结构化结果撰写长文
- **能力**：本地文件 + 知识库交叉引用，自动生成参考文献
- **输出**：Markdown（可编辑）和 PDF（正式提交）两种格式

---

## 三种工作模式

| 模式 | 说明 | 适用 |
|------|------|------|
| **Chat** | 标准 LLM 对话 | 日常问答 |
| **Reasoner** | 思维链推理 | 复杂分析 |
| **DeepDiver** | 万字长文生成 | 综述/报告/论文 |

DeepDiver 模式默认最大 40 轮迭代，适合生成需要大量文献支撑的完整报告。

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 模型后端 | 盘古 DeepDiver-V2 | 华为大模型，通过 litellm 接入 |
| LLM 网关 | litellm | 统一接口，可替换为 OpenAI/DeepSeek 等 |
| 前端 | Flask + REST + WebSocket | 实时推送任务进度 |
| 存储 | MySQL + 文件系统 | 会话管理、用户认证、文件库 |
| 搜索 | MCP 协议 | 接入 PubMed、ArXiv、Google Search |
| 多智能体 | 自研 MAS 框架 | 基于 ReAct 模式的 Agent 协作 |

---

## 代码结构

```
SciAssistant/
├── app.py           # Flask 主程序（62KB，含全部 Web 路由）
├── chatAi/          # 前端界面（HTML + 静态资源）
├── deepdiver_v2/    # 核心引擎
│   ├── src/agents/  # 三智能体实现
│   │   ├── planner_agent.py
│   │   ├── objective_information_seeker.py
│   │   ├── subjective_information_seeker.py
│   │   └── writer_agent.py
│   ├── src/tools/   # MCP 工具 / PDF 处理
│   └── config/      # 配置 + .env 模板
└── requirements.txt
```

---

## 与 DeepEarth / EVE 的对比

| 项目 | 领域 | 核心能力 | 接入方式 |
|------|------|---------|---------|
| TorchGeo | 遥感数据 | 数据集 + 采样器 | Python 库 |
| DeepEarth | 地球科学 AI | 时空编码 | Python 库 |
| EVE | 地球科学知识 | RAG 问答 | REST API / MCP |
| **SciAssistant** | **科研自动化** | **多智能体写作** | **Web 应用** |

四者覆盖了「数据处理 → 特征编码 → 知识检索 → 自动写作」的完整链路。

---

## 本地部署要点

1. MySQL 数据库（存储用户/会话/文件）
2. LLM 后端：配置 `MODEL_REQUEST_URL` 指向 OpenAI 兼容 API
3. Python 3.11+，`pip install -r requirements.txt`
4. 可选：MCP 搜索服务（用于网络检索）
5. 初始化数据库 → 启动 Flask → 浏览器访问

如果只是理解架构（不部署），不需要 MySQL——直接读 `deepdiver_v2/src/agents/` 下的代码即可。

---

## 参考

- [SciAssistant GitHub](https://github.com/scsio-marinebio/SciAssistant)
- [盘古 DeepDiver-V2](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-DeepDiver)
- [ReAct 论文](https://arxiv.org/abs/2210.03629) — 多智能体推理模式的理论基础
- [MCP 协议规范](https://modelcontextprotocol.io/)

---

*首次发布于 2026-05-20 · 掘金 / 知乎*
