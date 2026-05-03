# agent.py
import os
import sys
import logging
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

# Dynamically find the absolute path to this folder and the server script
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT_PATH = os.path.join(AGENT_DIR, "gmail_server.py")

# Use sys.executable to guarantee it uses your .venv Python
server_params = StdioServerParameters(
    command=sys.executable, 
    args=[SERVER_SCRIPT_PATH]
)

# Set up the MCP tool connection
connection_params = StdioConnectionParams(server_params=server_params)
gmail_tools = McpToolset(connection_params=connection_params)

# 3. Configure logging
logging.basicConfig(level=logging.INFO)

# 4. Initialize your ADK Agent
root_agent = Agent(
    name="email_assistant",
    model="gemini-2.5-flash",
    instruction="""You are a helpful assistant that manages the user's email. 
When asked about emails, use the tools provided to fetch unread messages and summarize them clearly.""",
    tools=[gmail_tools],
)