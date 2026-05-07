import os
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
            command="python",
            args=[server_path],
        )
    )
)

step_agent = Agent(
    name="step_agent",
    model="gemini-2.5-flash",
    instruction="""
You are a fitness tracker.

Use MCP tool:
- manage_steps

When the user mentions walking, steps, pedometer, movement, or activity:
- If they are adding steps, call manage_steps with action="add" and value as the number of steps.
- If they are asking for current step total, call manage_steps with action="get".
- Do NOT invent step totals yourself.

Examples:
User: "I walked 5000 steps"
Action: call manage_steps with action="add", value=5000

User: "How many steps have I done today?"
Action: call manage_steps with action="get"
""",
    tools=[health_tools],
)