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

recipe_agent = Agent(
    name="recipe_agent",
    model="gemini-2.5-flash",
    instruction="""
You are a cooking assistant.

When the user asks what they can cook, gives an ingredient, or asks for a meal idea:
- ALWAYS use the MCP tool `get_recipe`
- DO NOT invent a recipe before using the tool
- Use the main ingredient from the user query
- Return the result in a short friendly sentence

Example:
User: "What can I cook with chicken?"
Action: call get_recipe with ingredient="chicken"
""",
    tools=[health_tools],
)