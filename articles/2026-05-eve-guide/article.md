# EVE 入门：ESA 的地球科学 AI 助手

> ESA Φ-lab × Mistral AI × Trillium Technologies 联合出品。读完你会用 EVE 的 API 和 MCP 协议做地球科学智能问答。

---

## EVE 是什么？

**EVE（Earth Virtual Expert）** 是欧洲航天局（ESA）Φ-lab 推出的开源地球科学大语言模型系统。三句话概括：

- **模型**：基于 Mistral Small 3.2（24B 参数），128K 上下文窗口
- **知识**：RAG 检索增强，内置地球观测文献库
- **接口**：REST API + MCP 协议，可嵌入任意 AI 工具链

已服务 300+ 地球观测研究用户，2026 年 5 月在 EGU 大会正式发布。

---

## 核心概念

### 1. RAG 架构

EVE 不是裸 LLM——它用检索增强生成（RAG）回答地学问题：

```
用户提问 → 检索相关文献（k篇） → 拼接上下文 → Mistral 24B 生成回答
```

公开知识库包括 ESA 的地球观测论文、遥感教程、卫星数据文档。

### 2. 三种接入方式

| 方式 | 适用场景 | 依赖 |
|------|---------|------|
| Python 客户端 | 脚本、Jupyter、数据分析 | `pip install eve-api` |
| MCP 协议 | Claude Code / Cursor / AI 工具 | `pip install eve-mcp` |
| 直接 HTTP | 任何语言、无依赖 | `httpx` / `curl` |

---

## Python API 客户端

安装：

```bash
cd eve-api && pip install -e .
```

异步使用模式（自动管理 JWT 刷新）：

```python
from eve_api import EVEClient

async with EVEClient() as eve:
    await eve.login("user@example.com", "password")

    # 用户信息
    me = await eve.get("/users/me")

    # 列出公开知识库
    collections = await eve.get("/collections/public")

    # 创建对话
    conv = await eve.post("/conversations", json={"name": "遥感问答"})

    # 流式 RAG 查询
    async for event in eve.stream(
        f"/conversations/{conv['id']}/stream_messages",
        json={
            "query": "What is NDVI and how is it computed?",
            "public_collections": ["eve-public"],
            "k": 5,
        },
    ):
        if event["type"] == "token":
            print(event["content"], end="", flush=True)
```

- `k` 参数控制检索文档数。增大会得更多背景但增加延迟
- 支持私有知识库：上传自己的论文库，建立个性化问答系统

---

## MCP 协议集成

EVE 提供了 MCP（Model Context Protocol）服务器，让任何支援 MCP 的 AI 工具直接调用：

```json
// Claude Code settings.json 配置
{
  "mcpServers": {
    "eve": {
      "command": "eve-mcp",
      "env": {
        "EVE_USER_EMAIL": "your@email.com",
        "EVE_USER_PASSWORD": "your-password"
      }
    }
  }
}
```

### 五个 MCP 工具

| 工具 | 功能 | 使用场景 |
|------|------|---------|
| `query_eve` | RAG 问答 | 「NDVI 为什么用近红外波段？」|
| `list_eve_collections` | 列出知识库 | 「EVE 里有没有海洋遥感文献？」|
| `check_eve_health` | 健康检查 | 调试连接 |
| `extract_factuality_issues` | 分析 GEE 脚本假设 | 审查代码科学性 |
| `assess_factuality_issue` | 专家级评估 | 逐条评估科学前提 |

后两个工具特别有价值——它们不是简单的问答，而是对 Google Earth Engine Python 脚本进行科学假设审查，告诉你代码哪里可能存在地学概念上的偏差。

---

## EVE 的定位

从 AI 生态角度看 EVE 的位置：

```
基础模型层: Mistral Small 3.2 (24B)
     ↓
RAG 层:     EVE 检索 + 地球观测文献库
     ↓
接口层:     REST API / MCP 协议
     ↓
应用层:     Claude Code / Cursor / Jupyter / Web Chat
```

和 TorchGeo（数据+模型）和 DeepEarth（时空编码器）不同，EVE 是**知识层**——它让 AI 在回答地学问题时接入权威文献，减少幻觉。

---

## 注册与使用

1. 访问 [https://eve-chat.chat](https://eve-chat.chat) 注册账号
2. Web 界面直接体验
3. API 文档：[https://api.eve-chat.chat/docs](https://api.eve-chat.chat/docs)

---

## 参考

- [EVE-API GitHub](https://github.com/FrontierDevelopmentLab/eve-api)
- [EVE-MCP GitHub](https://github.com/FrontierDevelopmentLab/eve-mcp)
- [EVE HuggingFace 模型](https://huggingface.co/eve-esa/EVE-Instruct)
- [Trillium Technologies](https://trillium.tech/)
- [Earth System Predictability](https://eslab.ai/esp)

---

*首次发布于 2026-05-20 · 掘金 / 知乎*
