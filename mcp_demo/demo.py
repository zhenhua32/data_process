import asyncio
from fastmcp import Client, FastMCP
from fastmcp.client.auth import BearerAuth
from fastmcp.client.transports import StreamableHttpTransport

http_url = "https://mcp-api.zhidemai.com/mcp"  # HTTP server URL
token = "xxx"

config = {
    "mcpServers": {
        "remote": {
            "url": http_url,
            "headers": {
                "Authorization": f"Bearer {token}",
            },
        }
    }
}

# xxx 需要登录 https://mcp.higress.ai/server/server9016 获取
config = {"mcpServers": {"mcp-ip-query": {"url": "https://mcp.higress.ai/mcp-ip-query/xxx"}}}


async def main():
    # Connect via in-memory transport
    # async with Client(http_url, auth=token) as client:
    # async with Client(http_url, auth=BearerAuth(token=token),) as client:
    # async with Client(http_url) as client:
    async with Client(config) as client:
        # await client.ping()
        # ... use the client
        # Make MCP calls within the context
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        print("start")
        # result = await client.call_tool("fmi_content_search", {"query": "5070笔记本"})
        result = await client.call_tool("ip-address-query", {"ip": ""})
        print(f"get result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
