import os
import asyncio
from browser_use import Agent, Browser
from pydantic import SecretStr
from langchain_openai import AzureChatOpenAI
from browser_use.browser.context import BrowserContext, BrowserContextConfig

# Initialize the model
llm = AzureChatOpenAI(
    model="gpt-4.1-mini",
    api_version='2025-01-01-preview',
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
    api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
)
task = 'Go to realtor.ca and search for houses in Ottawa, go to the first listing and click on it and verify if the price is visible'

config = BrowserContextConfig(
    allowed_domains=['realtor.ca'],
)

browser = Browser()
context = BrowserContext(browser=browser, config=config)

agent = Agent(task=task, llm=llm, use_vision='False', browser_context=context, max_actions_per_step=2)


async def main():
	await agent.run(max_steps=15)


if __name__ == '__main__':
	asyncio.run(main())
