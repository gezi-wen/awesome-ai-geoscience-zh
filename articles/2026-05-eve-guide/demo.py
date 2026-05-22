"""
EVE (Earth Virtual Expert) 入门 —— ESA 地球科学 LLM
=====================================================
演示 EVE Python API 客户端 + MCP 协议集成。

EVE 是 ESA Φ-lab 联合 Mistral AI 打造的地球科学大语言模型，
基于 Mistral Small 3.2 (24B)，128K 上下文窗口，支持 RAG 检索。

运行前：
  cd /mnt/e/workspace/eve/eve-api
  pip install -e .

需要 EVE 账号: https://eve-chat.chat
"""

import asyncio

# ─── 方式 1：Python API 客户端 ──────────────────
# from eve_api import EVEClient

async def demo_api_client():
    """使用 eve-api Python 客户端（需要 EVE 账号）"""
    print("=" * 55)
    print("EVE · 方式 1：Python API 客户端")
    print("=" * 55)

    # EVEClient 是异步上下文管理器，自动管理 JWT 刷新
    # async with EVEClient() as eve:
    #     await eve.login("your@email.com", "password")
    #
    #     # 查询用户信息
    #     me = await eve.get("/users/me")
    #     print(f"认证用户: {me['email']}")
    #
    #     # 列出公开知识库
    #     collections = await eve.get("/collections/public")
    #     for col in collections["data"]:
    #         print(f"  📚 {col['name']} — {col.get('description', '')[:80]}")
    #
    #     # RAG 查询（流式）
    #     conv = await eve.post("/conversations", json={"name": "遥感问答"})
    #     async for event in eve.stream(
    #         f"/conversations/{conv['id']}/stream_messages",
    #         json={
    #             "query": "What is the role of AI in land cover classification?",
    #             "public_collections": ["eve-public"],
    #             "k": 5,  # 检索 top-5 文档
    #         },
    #     ):
    #         if event["type"] == "token":
    #             print(event["content"], end="", flush=True)
    #         elif event["type"] == "final":
    #             print()

    print("\n(需要 EVE 账号登录 — 注册地址: https://eve-chat.chat)\n")


# ─── 方式 2：MCP 协议集成 ─────────────────────
def demo_mcp_integration():
    """通过 MCP 协议调用 EVE（Claude Code / Cursor / 等 MCP 客户端）"""
    print("=" * 55)
    print("EVE · 方式 2：MCP 协议工具调用")
    print("=" * 55)

    print("""
配置 Claude Code settings.json:

{
  "mcpServers": {
    "eve": {
      "command": "eve-mcp",
      "args": [],
      "env": {
        "EVE_API_KEY": "your-api-key",
        "EVE_USER_EMAIL": "your@email.com",
        "EVE_USER_PASSWORD": "your-password"
      }
    }
  }
}

可用 MCP 工具:
  • query_eve          — RAG 查询地球科学知识
  • list_eve_collections — 列出公开知识库
  • check_eve_health   — API 健康检查
  • extract_factuality_issues — 分析 GEE 脚本中的科学假设
  • assess_factuality_issue   — 专家级评估
""")


# ─── 方式 3：直接用 HTTP ──────────────────────
async def demo_http():
    """直接 HTTP 调用（最小依赖）"""
    print("=" * 55)
    print("EVE · 方式 3：直接 HTTP 调用")
    print("=" * 55)

    print("""
import httpx

# 登录
resp = httpx.post("https://api.eve-chat.chat/v1/auth/login", json={
    "email": "your@email.com",
    "password": "your-password"
})
token = resp.json()["access_token"]

# 查询
headers = {"Authorization": f"Bearer {token}"}
resp = httpx.post("https://api.eve-chat.chat/v1/conversations", headers=headers,
    json={"name": "测试"})
conv_id = resp.json()["id"]

# 流式 RAG 回答
with httpx.stream("POST",
    f"https://api.eve-chat.chat/v1/conversations/{conv_id}/stream_messages",
    headers=headers,
    json={"query": "Explain NDVI in remote sensing", "k": 3}
) as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(line)
""")


async def main():
    await demo_api_client()
    demo_mcp_integration()
    await demo_http()
    print("=" * 55)
    print("✅ EVE 演示完成")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
