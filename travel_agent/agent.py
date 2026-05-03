from google.adk.agents.llm_agent import Agent

travel_agent = Agent(
    model="gemini-2.5-flash",
    name="travel_agent",
    description="Helps with travel planning, flights, hotels, and itineraries.",
    instruction="""
You are a travel assistant.

Help users with:
- Trip planning
- Flight suggestions
- Hotel recommendations
- Daily itineraries

Always respond in a clear travel guide format.
"""
)