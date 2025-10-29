import os

import crewai
import litellm
import pandas as pd
from crewai import LLM, Agent, Task, Crew, Process
from crewai.project import agent, task
from crewai.tools import tool, BaseTool
from crewai_tools import DirectoryReadTool, FileReadTool
from langchain_experimental.utilities import PythonREPL

from agents_sql import task_sql

# create sqlite db from Titanic dataset
dtf = pd.read_csv("data/data_titanic.csv")
print(dtf.head(3))

# llm = LLM(model="ollama/mannix/deepseek-coder-v2-lite-instruct:latest")
# llm = LLM(model="ollama/gemma3:4b")
# llm = LLM(model="ollama/codestral:latest")
llm = LLM(model="ollama/qwen3:8b")
# llm = LLM(model="ollama/chevalblanc/gpt-4o-mini")

# Python Data Analyst Agent & Task
prompt = '''Extract data with Python code to answer the question '{user_input}'
        '''


@tool("tool_dir", result_as_answer=False)
def tool_dir(directory: str) -> str:
    """
    Returns directory listing for given **directory** argument
    """
    return DirectoryReadTool(directory).run()
    return res


@tool("tool_file", result_as_answer=False)
def tool_file(file: str) -> str:
    """
    Reads the file provided as argument contents
    """
    return FileReadTool(file).run()


# Python code execution tool
@tool("tool_pycode", result_as_answer=True)
def tool_pycode(python_code: str) -> str:
    """
    Use this tool to execute a Python code. Input should be a valid python script.
    You should print the returned results using `print(...)` to the console.
    """
    print(f'!!! Executing: {python_code}\n')
    return PythonREPL().run(python_code)


# Python code evaluation and debugging tool
@tool("tool_eval", result_as_answer=False)
def tool_eval(python_code: str) -> str:
    """
    Use this tool before executing Python code to evaluate code and correct it if necessary.
    Example: `import numpy as np print(np.sum([1,2]))` would give an error,
    so you must change it to `import numpy as np; print(np.sum([1,2]))`
    """
    res = llm.call(messages='''Review the following python code and correct it if you find errors. Be sure to print the final result in the code.
    You must return very short answer in json format {"answer":<answer_as_text>, "code":<corrected_python_code>}:\n''' + python_code)
    print(f'!!! Validating: {python_code}\n !!! Corrected: {res}')
    return res


## Agent
data_analyst = crewai.Agent(
    role="Data Analyst",
    goal=prompt,
    backstory='''
        You are Python data analyst. Use ONLY the data provided to answer the {user_input}. To solve the problem follow the steps:
        1. Generate <python code> you need to load data from file and answer the question and log the code. Use ';' as line separator. Be sure to print the final result in the code.
        2. Use tool `tool_eval` to check and debug the generated <python code>.
        3. Use tool `tool_pycode` to execute the <python code> and return the output as <final answer>. 
        !IMPORTANT: ALWAYS use tool `tool_pycode` to generate the <final answer>.
        ''',
    tools=[tool_eval, tool_pycode],
    max_iter=30,
    llm=llm,
    allow_delegation=False,
    verbose=True
)

## Task
task_py = Task(
    description=prompt,
    agent=data_analyst,
    context=[task_sql],
    expected_output='''Output of `tool_pycode`.'''
)

if __name__ == '__main__':
    print(f'How many passengers in Titanic: {len(dtf)}')
    print(f'How many passengers survived: {len(dtf[dtf["Survived"] == 1])}')
    print(f'How many passengers did not survive: {len(dtf[dtf["Survived"] == 0])}')

    result = tool_pycode.run("import numpy as np; print(np.sum([1,2]))")
    print(f'\n{result}')

    # print(tool_eval.run("print(Res:')"))

    # Calling data-scinetist agent
    os.environ['LITELLM_LOG'] = 'DEBUG'
    crew = Crew(
        agents=[
            data_analyst,
        ],
        tasks=[
            task_py
        ],
        verbose=True
    )
    file = "./data/data_titanic.csv"
    res = crew.kickoff(inputs={
        "user_input": f"How many people died in provided dataset from file '{file}'? Return only a single number."
    })
    print(res)
