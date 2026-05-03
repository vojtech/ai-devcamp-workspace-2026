from google.adk.agents.llm_agent import Agent
import requests

def get_weather(city: str) -> str:
    """
    Simple working weather tool (FREE, no API key)
    Uses Open-Meteo with minimal logic.
    """

    # Step 1: very small built-in city mapping (only for demo stability)
    city = city.lower().strip()

    locations = {
        "london": (51.5072, -0.1276),
        "paris": (48.8566, 2.3522),
        "new york": (40.7128, -74.0060),
    }

    if city not in locations:
        return "❌ City not supported. Try: London, Paris, New York"

    lat, lon = locations[city]

    # Step 2: call weather API
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )

    data = requests.get(url).json()

    weather = data["current_weather"]

    return f"""
🌍 {city.title()}
🌡️ Temp: {weather['temperature']}°C
💨 Wind: {weather['windspeed']} km/h
"""

weather_agent = Agent(
    model="gemini-2.5-flash",
    name="weather_agent",
    description="Provides weather information and forecasts.",
    instruction="""
You are a weather assistant.

Help users with:
- Current weather
- Forecasts
- Best time to travel based on weather
- Packing suggestions

Keep answers short and practical.
""",
tools=[get_weather]
)