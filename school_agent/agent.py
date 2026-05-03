from google.adk.agents.llm_agent import Agent
from google.adk.tools import load_memory, preload_memory
from school_agent.agents.english_agent import english_agent
from school_agent.agents.math_agent import math_agent
from school_agent.agents.history_agent import history_agent
from school_agent.agents.science_agent import science_agent

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="Main router agent that sends tasks to sub-agents.",
    instruction="""
Act as school coordinator.

Users can ask questions about:
- English (grammar, writing, style)
- Math (concepts, problem solving)
- Science (explanations, experiments)
- History (events, facts)

Route questions to the appropriate sub-agent.
""",
    tools=[preload_memory, load_memory],
    sub_agents=[english_agent, math_agent, history_agent, science_agent]
)