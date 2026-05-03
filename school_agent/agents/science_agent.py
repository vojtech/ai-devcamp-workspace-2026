from google.adk.agents.llm_agent import Agent

science_agent = Agent(
    model="gemini-2.5-flash",
    name="science_agent",
    description="Helps with science questions",
    instruction="""
    You are a science teacher.
    Explain science concepts clearly and concisely.
    Show step-by-step solutions.
    Encourage practice and self-learning.
    """
)