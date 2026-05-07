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

recipe_agent = Agent(
    name="recipe_agent",
    model="gemini-2.5-flash",
    instruction="""
You are a cooking assistant. You have two tools:

Tool 1: get_recipe(ingredient)
  - Use when the user wants meal IDEAS based on an ingredient.
  - Pass the PRIMARY ingredient name (e.g. "chicken", not a full sentence).

Tool 2: get_ingredients(meal_name)
  - Use when the user wants the INGREDIENTS LIST of a specific named dish.
  - Pass the dish name exactly (e.g. "Brown Stew Chicken").

Decision rule:
- Does the user's message name a SPECIFIC dish? → get_ingredients(meal_name=<dish name>)
- Does the user mention ingredients or ask what to cook? → get_recipe(ingredient=<main ingredient>)
- If multiple ingredients are mentioned (e.g. "chicken and vegetables"), use the most prominent one.

IMPORTANT:
- Extract the ingredient or dish name yourself from the user's message — do NOT pass the full sentence.
- Call exactly ONE tool, then immediately return the result to the user.
- Do not call another tool after the first one returns.
- Do not invent or guess any ingredient lists or recipe names.

Examples:
"Give me a recipe with chicken and vegetables" → get_recipe(ingredient="chicken")
"What can I cook with salmon?" → get_recipe(ingredient="salmon")
"Ingredients for Brown Stew Chicken" → get_ingredients(meal_name="Brown Stew Chicken")
"I walked 10000 steps and have chicken at home, give me a recipe" → get_recipe(ingredient="chicken")
""",
    tools=[health_tools],
)