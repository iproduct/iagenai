import os

import crewai
import litellm
import pandas as pd
from crewai import LLM, Agent, Task, Crew, Process
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import agent, task, CrewBase
from crewai.tools import tool, BaseTool
from crewai_tools import DirectoryReadTool, FileReadTool
from langchain_experimental.utilities import PythonREPL

from agents_sql import  task_sql

# create sqlite db from Titanic dataset
dtf = pd.read_csv("data/data_titanic.csv")
print(dtf.head(3))


# llm = LLM(model="ollama/mannix/deepseek-coder-v2-lite-instruct:latest")
# llm = LLM(model="ollama/gemma3:4b")
# llm = LLM(model="ollama/codestral:latest")
llm = LLM(model="ollama/qwen3:8b")
# llm = LLM(model="ollama/chevalblanc/gpt-4o-mini")

# Python Data Analyst Agent & Task
prompt = '''Extract data with Python code to answer the question '{question}', by loading the dataset from file '{file}' in directory '{directory}' using appropriate tools.
        IMPORTANT: Think verbosely. Generate and execute python code to answer the question. Use ONLY data from file '{file}' in directory '{directory}. Use ALL provided tools. Follow these steps:
        FIRST generate <python code> you need to load dataset from file and answer the question;
        THEN use tool `tool_eval` to check and debug the generated <python code>;
        FINALLY use tool `tool_pycode` to execute the <python code> and return the output as <final answer>. 
        ONLY return <final answer> AFTER calling <tool> `tool_pycode`.
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
    A Python shell. Use this to execute python commands. Input should be a valid python command.
    If you want to see the output of a value, you should print it out with `print(...)`.
    """
    print(f'!!! Executing: {python_code}\n')
    return PythonREPL().run(python_code)

# Python code evaluation and debugging tool
@tool("tool_eval", result_as_answer=False)
def tool_eval(python_code: str) -> str:
    """
    Before executing Python code, always use this tool to evaluate code and correct the code if necessary.
    Example: `import numpy as np print(np.sum([1,2]))` would give an error,
    so you must change it to `import numpy as np; print(np.sum([1,2]))`
    """
    res = llm.call(messages='''review the following python code and correct it if you find errors.
    You must return very short answer in json format {"answer":<answer_as_text>, "code":<corrected_python_code>}:\n''' + python_code)
    print(f'!!! Validating: {python_code}\n !!! Corrected: {res}')
    return res

## Agent
data_analyst = crewai.Agent(
        role="Data Analyst",
        goal=prompt,
        backstory='''
        You are Python data analyst. Use ONLY the data provided in {file} to answer the {question}.
        IMPORTANT: Think verbosely. Use ALL provided <tools>. Follow the steps:
        FIRST generate <python code> you need to load data from file and answer the question and log the code;
        THEN use <tool> `tool_eval` to check and debug the generated <python code>;
        FINALLY use <tool> `tool_pycode` to execute the <python code> and return the output as <final answer>. 
        ONLY return <final answer> AFTER calling <tool> `tool_pycode`.
                ''',
        tools=[tool_dir, tool_file, tool_eval, tool_pycode],
        max_iter=30,
        llm=llm,
        allow_delegation=False,
        verbose=True
    )


## Task
@task
def question_answer_task() -> Task:
    return Task(
        description=prompt,
        agent=data_analyst,
        context=[task_sql],
        expected_output='''Output of `tool_pycode`.'''
    )

task_py = question_answer_task()

# task_py = crewai.Task(
#     description=prompt,
#     agent=data_analyst,
#     # context=[task_sql],
#     expected_output='''Output of TOOL `tool_pycode`.''')

if __name__ == '__main__':
    print(f'How many passengers in Titanic: {len(dtf)}')
    print(f'How many passengers survived: {len(dtf[dtf["Survived"] == 1])}')
    print(f'How many passengers did not survive: {len(dtf[dtf["Survived"] == 0])}')

    result = tool_pycode.run("import numpy as np; print(np.sum([1,2]))")
    print(f'\n{result}')

    print(tool_eval.run("print(Res:')"))

    # Calling data-scinetist agent
    os.environ['LITELLM_LOG'] = 'DEBUG'
    crew = Crew(
        agents=[
            agent,
        ],
        tasks=[
            task_py
        ],
        process=Process.sequential,
        verbose = True
    )
    res = crew.kickoff(inputs={
        "question": f"How many people died in provided dataset? Return only a single number.",
        "file": "data_titanic.csv",
        "directory": "./data"
    })
    print(res)
