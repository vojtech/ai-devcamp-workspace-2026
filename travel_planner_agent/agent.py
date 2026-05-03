from google.adk.agents.llm_agent import Agent
#from google.adk.tools import google_search
from datetime import datetime
from travel_agent.agent import travel_agent
from weather_agent.agent import weather_agent
from hotel_agent.agent import hotel_agent
from flight_agent.agent import flight_agent
from itinerary_agent.agent import itinerary_agent

#def now() -> dict:
#    """Returns the current date and time."""
#    my_datetime = datetime.now()
#    return {
#        "status": "success",
#        "current_time": str(my_datetime)
#    }

#root_agent = Agent(
#    model='gemini-2.5-flash',
#    name='root_agent',
#    description='A helpful assistant for user questions.',
#    instruction='Answer user questions to the best of your knowledge',
#    sub_agents=[travel_agent],


root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="Main travel planning agent that coordinates flights, stays, and itineraries.",
    instruction="""
You are a helpful travel agent.

Your job is to help the user plan trips, find flights, Airbnb stays, and create itineraries.

Available specialists:
- flight_agent for flights
- hotel_agent for Airbnb stays
- itinerary_agent for trip plans

Rules:
- Delegate to the correct specialist based on the user's request.
- If the user asks for a complete trip, help with flights, stay, and itinerary.
- If the user has not provided enough details, ask one short follow-up question.
- Keep answers simple and helpful.
- When finished, reply with "DONE".
""",
    sub_agents=[travel_agent, weather_agent, hotel_agent, flight_agent, itinerary_agent]
)
