import crewai
import pandas as pd
from crewai import LLM
from crewai.tools import tool
from langchain_ollama import OllamaLLM
import sqlite3
from langchain_community.tools import ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDatabaseTool, QuerySQLCheckerTool
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

# Python code execution tool
tool_pycode = Tool(name="tool_pycode",
    description='''
    A Python shell. Use this to execute python commands. Input should be a valid python command. 
    If you want to see the output of a value, you should print it out with `print(...)`.
    ''',
    func=PythonREPL().run)

@tool("tool_eval")
def tool_eval(code: str) -> str:
    """
    Before executing Python code, always use this tool to evaluate code and correct the code if necessary.
    Example: `import numpy as np print(np.sum([1,2]))` would give an error,
    so you must change it to `import numpy as np; print(np.sum([1,2]))`
    """
    res = llm.invoke(input=['''review the following python code and correct it if you find errors.
    You must return very short answer in json format {"answer":<answer>, "code":<corrected code>}:\n''' + code]).split("\n")
    return res



if __name__ == '__main__':
    dtf = pd.read_csv("data_titanic.csv")
    print(dtf.head(3))
    llm = OllamaLLM(model="qwen3:8b")
    # crewai_llm = LLM(model="ollama/codestral:latest ")
    crewai_llm = LLM(model="ollama/qwen3:8b")

    print(f'How many passengers in Titanic: {len(dtf)}')
    print(f'How many passengers survived: {len(dtf[dtf["Survived"] == 1])}')
    print(f'How many passengers did not survive: {len(dtf[dtf["Survived"] == 0])}')

    result = tool_pycode.run("import numpy as np; print(np.sum([1,2]))")
    print(f'\n{result}')

    print(tool_eval.run("print(Res:')"))