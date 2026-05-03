"""
Travel Agent module using Google ADK.
This module defines a travel agent with Airbnb access via MCP.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


def now() -> dict:
    """Returns the current date and time in Europe/London."""
    current_dt = datetime.now(ZoneInfo("Europe/London"))
    return {
        "status": "success",
        "current_time": current_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


airbnb_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@openbnb/mcp-server-airbnb@0.1.2",
                "--ignore-robots-txt",
            ],
        ),
        timeout=30,
    )
)


hotel_agent = Agent(
    name="hotel_agent",
    model="gemini-2.5-flash",
    tools=[now, airbnb_mcp],
    instruction="""
You are a helpful hotel agent.

Your job is to help the user plan trips and find Airbnb stays.

Available tools:
- Use the Airbnb MCP tools to search for accommodation and pricing.
- Use the now tool only when the user asks for the current date or time.

Rules:
- Do not use Google Search.
- If the user has not provided enough details, ask one short follow-up question.
- For stay suggestions, include location, approximate cost, and why it matches.
- When finished, reply with "DONE".
""",
)