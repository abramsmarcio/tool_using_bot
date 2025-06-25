import os
import requests
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=1024,
    timeout=30,
)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city using OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"The weather in {city.title()} is {desc} with a temperature of {temp}°C."
    else:
        return f"Weather data not found for {city.title()}."


@tool
def mul(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


llm_with_tools = llm.bind_tools([get_weather, mul])


messages = [
    SystemMessage(content="You are a helpful assistant. Use tools only when the user explicitly asks for weather or math help.")
]

print("Start chatting (type 'exit' to quit):")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting. Bye!")
        break
    messages.append(HumanMessage(content=user_input))
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    tool_calls = ai_msg.additional_kwargs.get("tool_calls", [])
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            print(f"Tool called: {tool_name} with args: {tool_args}")

            if tool_name == "get_weather":
                result = get_weather.invoke(tool_args)
            elif tool_name == "mul":
                result = mul.invoke(tool_args)
            else:
                result = "Tool not recognized."


            
            messages.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=str(result)
                )
            )

    
        final_response = llm_with_tools.invoke(messages)
        messages.append(final_response)
        print("Bot:", final_response.content)
    else:
        print("Bot:", ai_msg.content)
