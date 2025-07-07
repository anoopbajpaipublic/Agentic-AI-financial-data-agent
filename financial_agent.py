from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
from dotenv import load_dotenv
import os

openai.api_key = os.getenv("OPENAI_API_KEY")



#This is my web search agent that can search the web for information and summarize it.
websearch_agent = Agent(
	 name="Web Search Agent",
     role="Search the web for information",
     model=Groq(id="llama3-8b-8192"),
    tools=[DuckDuckGo()],
    instructions="Always include the source of the information you find. If you cannot find the information, say so. If you find multiple sources, summarize the most relevant ones.",
    markdown=True,
)

#financial agent 
financial_agent = Agent(
    name="Finance AI Agent",
    role="Analyze financial data and provide insights",
    model=Groq(id="llama3-8b-8192"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions="Use tables to display the data, Provide a summary of the financial health of the company, and include any relevant news or analyst recommendations. If you cannot find the information, say so.",
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=  [websearch_agent, financial_agent],
    instructions="always include the source and use tables to display the data",
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream=True)
    