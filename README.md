# Awesome AI Geoscience 中文技术写作

AI + 地球科学交叉领域的中文技术文章合集。面向遥感、时空模型、多智能体科研系统的入门与深度解析。

## 文章目录

### 入门教程

| 文章 | 简介 | demo | 发布 |
|------|------|------|------|
| [TorchGeo 入门](articles/2026-05-torchgeo-guide.md) | 用 PyTorch 处理遥感数据，EuroSAT 分类实战 | ✅ | [掘金] |
| [Prithvi 遥感基础模型](articles/2026-05-prithvi-guide.md) | NASA×IBM 的遥感 Vision Transformer | ✅ | 待发 |

### 架构解析

| 文章 | 简介 | demo | 发布 |
|------|------|------|------|
| [DeepEarth 深度解析](articles/2026-05-deepearth-guide.md) | Earth4D 时空编码器工作原理 | ✅ | 待发 |
| [GISclaw LLM 空间分析](articles/2026-05-gisclaw-guide.md) | 纯 Python 开源 GIS Agent 系统 | ✅ | 待发 |

### 工具与生态

| 文章 | 简介 |
|------|------|
| [EVE 入门](articles/2026-05-eve-guide.md) | ESA 地球科学 AI 助手，RAG + MCP 协议 |
| [SciAssistant 架构解析](articles/2026-05-sciassistant-guide.md) | 中科院南海所 Multi-Agent 科研写作系统 |
| [AI+地学技术栈总览](articles/2026-05-ai-geo-stack.md) | 从卫星图像到科研报告的完整链路 |

### 方法论

| 文章 | 简介 |
|------|------|
| [MCP 协议与地学工具链](articles/2026-05-mcp-geo.md) | 用 MCP 串联四个地学 AI 项目 |
| [AI 地学技术写作指南](articles/2026-05-getting-started.md) | 从入门到变现的完整路线 |

## 开源贡献

- [DeepEarth — 中文 README 翻译](https://github.com/legel/deepearth/pulls)
- [SciAssistant — SQLite 后端适配器 + 多 LLM 部署指南](https://github.com/scsio-marinebio/SciAssistant/pull/4)

## 验证环境

所有 demo 在以下环境实测通过：

- **GPU**: NVIDIA RTX 4060 Laptop (8GB), CUDA 13.2
- **Python**: 3.12 + PyTorch 2.12
- **LLM**: DeepSeek V4 Pro
- **OS**: Ubuntu 22.04 / WSL2

## 许可证

MIT
