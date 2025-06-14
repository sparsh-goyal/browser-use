import os
import asyncio
import sys
from browser_use import Agent, Browser
from pydantic import SecretStr
from langchain_openai import AzureChatOpenAI
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from pathlib import Path

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

initial_actions = [
	{'open_tab': {'url': 'https://www.realtor.ca'}},
]

SCRIPT_DIR = Path('./playwright_scripts')
SCRIPT_PATH = SCRIPT_DIR / 'realtor_playwright_script.py'

agent = Agent(task=task, 
              llm=llm, 
              use_vision='False', 
              save_conversation_path = './logs/output.txt', 
              browser_context=context, max_actions_per_step=2, 	#initial_actions=initial_actions,
              save_playwright_script_path=str(SCRIPT_PATH),
)

# Helper function to stream output from the subprocess
async def stream_output(stream, prefix):
	if stream is None:
		print(f'{prefix}: (No stream available)')
		return
	while True:
		line = await stream.readline()
		if not line:
			break
		print(f'{prefix}: {line.decode().rstrip()}', flush=True)

def getHistory(history):
    
    if history.is_done():
        print("Model Thoughts:", history.model_thoughts())
        print("Visited URLs:", history.urls())
        # print("Screenshots:", history.screenshots())
        print("Executed Action Names:", history.action_names())
        print("Model Actions:", history.model_actions())
        print("Extracted Content:", history.extracted_content())
        print("Final Extracted Content:", history.final_result()) #final extracted content
        print("Action Results:", history.action_results())
        if history.has_errors():
            print("Errors?:", history.errors())
   
async def runGeneratedPlaywrightScript():
    print(f'\nChecking if Playwright script was generated at: {SCRIPT_PATH}')
    if SCRIPT_PATH.exists():
        print('Playwright script found. Attempting to execute...')
        try:
            # Ensure the script directory exists before running
            SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

            # Execute the generated script using asyncio.create_subprocess_exec
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(SCRIPT_PATH),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd(),  # Run from the current working directory
            )

            print('\n--- Playwright Script Execution ---')
            # Create tasks to stream stdout and stderr concurrently
            stdout_task = asyncio.create_task(stream_output(process.stdout, 'stdout'))
            stderr_task = asyncio.create_task(stream_output(process.stderr, 'stderr'))

            # Wait for both stream tasks and the process to finish
            await asyncio.gather(stdout_task, stderr_task)
            returncode = await process.wait()
            print('-------------------------------------')

            if returncode == 0:
                print('\n✅ Playwright script executed successfully!')
            else:
                print(f'\n⚠️ Playwright script finished with exit code {returncode}.')

        except Exception as e:
            print(f'\n❌ An error occurred while executing the Playwright script: {e}')
    else:
        print(f'\n❌ Playwright script not found at {SCRIPT_PATH}. Generation might have failed.')

    # Close the browser used by the agent (if not already closed by agent.run error handling)
    # Note: The generated script manages its own browser instance.
    if browser:
        await browser.close()
        print("Agent's browser closed.") 
        
async def main():
    try:
        history = await agent.run(max_steps=15)
        print('Agent finished running.')

        if history and history.is_successful():
            print(f'Agent completed the task successfully. Final result: {history.final_result()}')
            # getHistory(history)
        elif history:
            print('Agent finished, but the task might not be fully successful.')
            if history.has_errors():
                print(f'Errors encountered: {history.errors()}')
        else:
            print('Agent run did not return a history object.')

    except Exception as e:
        print(f'An error occurred during the agent run: {e}')
        # Ensure browser is closed even if agent run fails
        if browser:
            await browser.close()
        return  # Exit if agent failed

    await runGeneratedPlaywrightScript()

if __name__ == '__main__':
	asyncio.run(main())