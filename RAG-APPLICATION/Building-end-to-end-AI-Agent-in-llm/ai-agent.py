import os
import requests

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


# ================================
# Environment variables required
# ================================
# export OPENAI_API_KEY="your-openai-key"
# export WEATHERSTACK_API_KEY="your-weatherstack-key"


# ================================
# Tool 1: DuckDuckGo Search
# ================================
search_tool = DuckDuckGoSearchRun()


# ================================
# Tool 2: Weather Tool
# ================================
@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city.
    """
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Weather API key not found."

    url = (
        f"http://api.weatherstack.com/current"
        f"?access_key={api_key}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather for {city}"

    weather_desc = data["current"]["weather_descriptions"][0]
    temperature = data["current"]["temperature"]

    return (
        f"The current weather in {city} is {weather_desc} "
        f"with a temperature of {temperature}°C."
    )


# ================================
# LLM
# ================================
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini"  # or gpt-4.1 / gpt-3.5-turbo
)


# ================================
# ReAct Prompt
# ================================
prompt = hub.pull("hwchase17/react")


# ================================
# Create Agent
# ================================
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)


# ================================
# Agent Executor
# ================================
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)


# ================================
# Run Agent
# ================================
if __name__ == "__main__":
    query = (
        "Find the capital of Madhya Pradesh, "
        "then find its current weather condition"
    )

    result = agent_executor.invoke({"input": query})
    print("\nFinal Output:\n")
    print(result["output"])
