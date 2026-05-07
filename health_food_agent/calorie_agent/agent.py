import os
import sys
from google.adk import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    StdioServerParameters,
)

server_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "mcp_server",
    "mcp_health_server.py",
)

health_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[server_path],
        )
    )
)

calorie_agent = Agent(
    name="calorie_agent",
    model="gemini-2.5-flash",
    instruction="""
You are a nutrition expert. You have one tool: get_calories(food).

- Call get_calories with the food the user mentioned.
- After the tool returns, immediately give the result to the user. Do not call any other tool.
- Do NOT guess calories yourself.

Example:
User: "How many calories in paneer?"
Action: call get_calories with food="paneer", then return the result.
""",
    tools=[health_tools],
)