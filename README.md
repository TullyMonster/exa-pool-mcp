# Exa Pool MCP

轻量的 MCP Server，将 [Exa Pool API](https://github.com/chengtx809/exa-pool) 封装为可供 AI 助手调用的工具集。

## 安装与配置

### 1. 下载

下载 `exa_pool_mcp.py` 到本地：

```bash
# 下载到 ~/.claude/ 目录（推荐）
curl -o ~/.claude/exa_pool_mcp.py https://raw.githubusercontent.com/TullyMonster/exa-pool-mcp/main/exa_pool_mcp.py

# 或克隆整个仓库
git clone https://github.com/TullyMonster/exa-pool-mcp.git && cd exa-pool-mcp
```

### 2. 配置

> 以 Claude Code 为例：

```bash
claude mcp add --transport stdio exa-pool --env EXA_POOL_BASE_URL=... --env EXA_POOL_API_KEY=... -- uv run ~/.claude/exa_pool_mcp.py
```

重启 Claude Code 以应用 MCP 的配置变更。

## ❤️ 致谢与参考

- [Exa Pool GitHub](https://github.com/chengtx809/exa-pool)
- [Exa Docs](https://docs.exa.ai/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP](https://github.com/jlowin/fastmcp)

若需高使用限额或商业用途，请考虑使用 [Exa 官方 API](https://exa.ai/)。
