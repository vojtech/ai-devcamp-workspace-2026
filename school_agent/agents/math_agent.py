from google.adk.agents.llm_agent import Agent

math_agent = Agent(
    model="gemini-2.5-flash",
    name="math_agent",
    description="Helps with math questions",
    instruction="""
    You are a math teacher.
    Explain math concepts clearly and concisely.
    Show step-by-step solutions.
    Encourage practice and self-learning.
    """
)