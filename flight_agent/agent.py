from google.adk.agents import Agent
from urllib.parse import quote_plus

def search_flights_simple(origin: str, destination: str, date: str) -> dict:
    """Returns a Google Flights search link."""
    query = f"Flights from {origin} to {destination} on {date}"
    url = f"https://www.google.com/travel/flights?q={quote_plus(query)}"
    return {
        "status": "success",
        "message": "Here are available flight options.",
        "origin": origin,
        "destination": destination,
        "date": date,
        "booking_url": url,
    }


flight_agent = Agent(
    name="flight_agent",
    model="gemini-2.5-flash",
    description="Handles flight searches and returns Google Flights links.",
    tools=[search_flights_simple],
    instruction="""
You are a helpful flight booking assistant.

Your job is to help users find flights.

Rules:
- Use search_flights_simple when the user asks for flights.
- If origin, destination, or travel date is missing, ask one short follow-up question.
- Always include the booking link in your response.
- Keep the answer short and practical.
""",
)