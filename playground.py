import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app 
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")



#This is my web search agent that can search the web for information and summarize it.
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-8b-8192"),
    tools=[DuckDuckGo()],
    instructions=[
        "Always include the source of the information you find.",
        "If you cannot find the information, say so.",
        "If you find multiple sources, summarize the most relevant ones."
    ],
    markdown=True,
)


#financial agent 
financial_agent = Agent(
    name="Finance AI Agent",
    role="Analyze financial data and provide insights",
    model=Groq(id="llama3-8b-8192"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=[
        "Use tables to display the data.",
        "Provide a summary of the financial health of the company.",
        "Include any relevant news or analyst recommendations.",
        "If you cannot find the information, say so."
    ],
    show_tool_calls=True,
    markdown=True,
)


app= Playground(agents=[websearch_agent, financial_agent]).get_app()
if __name__ == "__main__": serve_playground_app("playground:app", reload=True)