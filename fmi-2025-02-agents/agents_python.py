import os

import crewai
import pandas as pd
from crewai import LLM
from crewai.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
import sqlite3
from langchain_community.tools import ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDatabaseTool, QuerySQLCheckerTool
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool
from pywin.framework.toolmenu import tools

crewai_llm = LLM(model="ollama/mannix/deepseek-coder-v2-lite-instruct:latest")
# crewai_llm = LLM(model="ollama/gemma3:4b")
# crewai_llm = LLM(model="ollama/codestral:latest")
# crewai_llm = LLM(model="ollama/qwen3:8b")

# create sqlite db from Titanic dataset
dtf = pd.read_csv("data_titanic.csv")
print(dtf.head(3))

# Python code execution tool
@tool("tool_pycode")
def tool_pycode(code: str) -> str:
    """
    A Python shell. Use this to execute python commands. Input should be a valid python command. 
    If you want to see the output of a value, you should print it out with `print(...)`.
    """
    return PythonREPL().run(code)

@tool("tool_eval")
def tool_eval(code: str) -> str:
    """
    Before executing Python code, always use this tool to evaluate code and correct the code if necessary.
    Example: `import numpy as np print(np.sum([1,2]))` would give an error,
    so you must change it to `import numpy as np; print(np.sum([1,2]))`
    """
    res = crewai_llm.call(messages='''review the following python code and correct it if you find errors.
    You must return very short answer in json format {"answer":<answer_as_text>, "code":<corrected_python_code>}:\n''' + code).split("\n")
    return res

# Python Data Analyst Agent & Task
prompt = '''Answer the question: "{question}", by following the steps: 
1) generating *python code* you need to analyze the data; 
2) using the `tool_eval` to check and debug the *python code*;
3) finally using `tool_pycode` to execute the *python code* and return the output.
The dataset to analyse is:\n{data}
'''

## Agent
agent_py = crewai.Agent(
    role="Data Analyst",
    goal=prompt,
    backstory='''
        You are an experienced data analyst that analyzes datasets using Python.
        You have attention to detail and always produce very clear and detailed results.
        First generate python code you need to analyze the {data} provided as part of the input.
        Then use `tool_eval` to check and debug generated python code.
        Finally use `tool_pycode` to execute the python code and return the output.
    ''',
    tools=[tool_eval, tool_pycode],
    max_iter=10,
    llm=crewai_llm,
    allow_delegation=False, verbose=True)

## Task
task_py = crewai.Task(
    description=prompt,
    agent=agent_py,
    # context=[task_sql],
    tools = [tool_eval, tool_pycode],
    expected_output='''Output of Python code''')
#%% md
##### Test single Agent



if __name__ == '__main__':
    print(f'How many passengers in Titanic: {len(dtf)}')
    print(f'How many passengers survived: {len(dtf[dtf["Survived"] == 1])}')
    print(f'How many passengers did not survive: {len(dtf[dtf["Survived"] == 0])}')

    result = tool_pycode.run("import numpy as np; print(np.sum([1,2]))")
    print(f'\n{result}')

    # print(tool_eval.run("print(Res:')"))

    # Calling data-scinetist agent
    os.environ['LITELLM_LOG'] = 'DEBUG'  # litellm.set_verbose=True
    crew = crewai.Crew(agents=[agent_py], tasks=[task_py], manager_llm = crewai_llm, chat_llm = crewai_llm, planning=False, verbose=True)
    res = crew.kickoff(inputs={
        "question": f"How many people died in the provided dataset?",
        "data": dtf.to_csv()
    })
    print(res)