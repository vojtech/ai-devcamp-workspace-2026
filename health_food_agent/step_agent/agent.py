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

step_agent = Agent(
    name="step_agent",
    model="gemini-2.5-flash",
    instruction="""
You are a fitness tracker. You have one tool: manage_steps(action, value).

Actions:
- action="add", value=N : user says they walked N steps
- action="get"          : user asks how many steps they have done
- action="reset"        : user wants to reset the step count

Call manage_steps once, then immediately return the result. Do not call any other tool.
Do NOT invent step counts yourself.

Examples:
User: "I walked 5000 steps" → manage_steps(action="add", value=5000)
User: "How many steps today?" → manage_steps(action="get")
""",
    tools=[health_tools],
)