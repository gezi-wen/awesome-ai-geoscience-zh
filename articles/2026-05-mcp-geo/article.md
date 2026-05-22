# MCP 协议：AI 地学工具链的通用胶水

> 理解 Model Context Protocol 如何串联 TorchGeo、DeepEarth、EVE 和 SciAssistant，打通数据到报告的完整链路。

---

## 什么是 MCP？

**MCP（Model Context Protocol）** 是 Anthropic 发布的开放协议，让 AI 模型安全地调用外部工具和数据源。简单理解：它是 AI 的「USB 接口」——任何实现了 MCP 的工具，AI 都能直接使用。

一个 MCP Server 暴露一组工具（Tools），AI 通过标准化的 JSON-RPC 协议调用它们：

```
AI 模型 ←→ MCP Client ←→ MCP Server ←→ 外部工具/数据
```

---

## 为什么地学 AI 需要 MCP？

地学 AI 项目有一个天然困境：**数据格式不统一、工具分散、接口各不相同**。

- TorchGeo 是 Python 库，需要通过代码调用
- DeepEarth 是 CUDA 模型，需要特定环境
- EVE 是 REST API，需要 HTTP 认证
- SciAssistant 是 Web 应用，需要浏览器交互

MCP 提供的价值：**把所有这些不同的接口统一成一套工具调用格式**。AI 不需要知道底层是 Python 库还是 REST API——它只需要知道「有个工具可以做这个」。

---

## 地学 AI 的 MCP 生态

### EVE-MCP：地球科学知识的 MCP 入口

EVE 是第一个为地学 AI 提供正式 MCP 支持的项目：

```json
// Claude Code settings.json
{
  "mcpServers": {
    "eve": {
      "command": "eve-mcp",
      "env": {
        "EVE_USER_EMAIL": "you@example.com",
        "EVE_USER_PASSWORD": "your-password"
      }
    }
  }
}
```

配置后，AI 可以直接使用五个工具：

| 工具 | 功能 |
|------|------|
| `query_eve` | RAG 查询地球科学文献 |
| `list_eve_collections` | 浏览可用知识库 |
| `check_eve_health` | 验证 API 连接 |
| `extract_factuality_issues` | 审查 GEE 代码的科学假设 |
| `assess_factuality_issue` | 专家级逐条评估 |

后两个工具尤其精彩——它们不只是「问答」，而是对 Google Earth Engine Python 脚本做科学审查，指出代码中的地学假设可能在哪里出错。

### SciAssistant：MCP 驱动的搜索

SciAssistant 内置了 MCP Server，通过 MCP 协议连接外部搜索引擎：

```
SciAssistant → MCP Client → MCP Server
                              ├── PubMed API
                              ├── ArXiv API
                              └── Google Search API
```

Info Seeker Agent 不需要知道每个搜索 API 的细节——它只需调用 MCP 工具，MCP Server 负责路由到正确的后端。

---

## 理想的地学 AI MCP 架构

如果将 TorchGeo、DeepEarth、EVE、SciAssistant 全部封装为 MCP Server：

```
AI (Claude / GPT)
  │
  ├── MCP: torchgeo-server
  │   ├── list_datasets()
  │   ├── load_dataset(name)
  │   └── sample_patches(bbox, size)
  │
  ├── MCP: deepearth-server
  │   ├── encode_spacetime(lat, lon, elev, time)
  │   └── predict_lfmc(region, date)
  │
  ├── MCP: eve-server
  │   ├── query_rag(question, collections)
  │   ├── list_collections()
  │   └── analyze_gee_script(code)
  │
  └── MCP: sciassistant-server
      ├── plan_research(topic)
      ├── search_literature(query)
      └── generate_report(outline, sources)
```

**AI 无需知道每个工具的底层实现**——它只需要知道「encode_spacetime 可以生成时空特征」，然后自由组合这些工具完成复杂任务。

---

## 实际工作流示例

> 用户：「分析 2023 年加州森林火灾风险趋势，写一份报告」

AI 通过 MCP 自动编排：

```
1. torchgeo-server.load_dataset("BigEarthNet")
   → 加载加州区域的卫星图像
   
2. deepearth-server.encode_spacetime(lat, lon, elev, "2023-07-01")
   → 生成 2023年7月 加州的时空特征向量
   
3. deepearth-server.predict_lfmc(region, date_range)
   → 预测活体燃料含水量 (火灾风险核心指标)
   
4. eve-server.query_rag("California wildfire trends 2023", ["eve-public"])
   → 检索 ESA 文献库中关于加州野火的最新研究
   
5. sciassistant-server.generate_report(outline, sources)
   → 整合分析结果 + 文献综述 → 生成报告
```

**5 个 MCP 调用，串联 4 个独立项目，完成一个完整的地学分析任务。**

---

## 当前的局限和机会

**现状**：只有 EVE 有正式的 MCP Server。TorchGeo、DeepEarth、SciAssistant 都只有 Python API 或 Web 界面。

**机会**：为这三个项目编写 MCP Server 封装，是极好的开源贡献方向：
- **TorchGeo MCP Server**：暴露数据集列表、加载、采样工具
- **DeepEarth MCP Server**：暴露时空编码、预测工具
- **SciAssistant MCP Server**：暴露文献检索、报告生成工具

这正是技术写作可以切入的角度——写教程的同时贡献代码。

---

## 参考

- [MCP 协议规范](https://modelcontextprotocol.io/)
- [EVE-MCP](https://github.com/FrontierDevelopmentLab/eve-mcp) — 首个地学 MCP 实现
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

---

*首次发布于 2026-05-20 · 掘金 / 知乎*
