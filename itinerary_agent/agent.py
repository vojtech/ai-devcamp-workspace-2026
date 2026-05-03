from google.adk.agents import Agent
from datetime import datetime
from zoneinfo import ZoneInfo

def now() -> dict:
    """Returns the current date and time in Europe/London."""
    current_dt = datetime.now(ZoneInfo("Europe/London"))
    return {
        "status": "success",
        "current_time": current_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

itinerary_agent = Agent(
    name="itinerary_agent",
    model="gemini-2.5-flash",
    description="Creates day-by-day travel itineraries based on destination, trip length, and interests.",
    tools=[now],
    instruction="""
You are a helpful itinerary planner.

Your job is to create practical travel itineraries.

Rules:
- Build a clear day-by-day itinerary.
- Use morning / afternoon / evening structure when useful.
- Group nearby places together to keep the plan realistic.
- Suggest food, sightseeing, and local experiences when relevant.
- If destination, trip length, or interests are missing, ask one short follow-up question.
- Use the now tool only if the user asks for the current date or time.
- Keep the itinerary easy to read.
""",
)