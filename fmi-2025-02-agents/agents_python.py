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

llm = OllamaLLM(model="qwen3:8b")
# crewai_llm = LLM(model="ollama/codestral:latest ")
crewai_llm = LLM(model="ollama/qwen3:8b")

# create sqlite db from Titanic dataset
dtf = pd.read_csv("data_titanic.csv")
print(dtf.head(3))
dtf.to_sql(index=False, name='titanic', con=sqlite3.connect("titanic.db"), if_exists='replace')

db = SQLDatabase.from_uri("sqlite:///titanic.db")


@tool("tool_tables")
def tool_tables() -> str:
    """Get all tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke('')


@tool("tool_schema")
def tool_schema(tables: str) -> str:
    """Get tables schema. Example input: table1, table2, table3"""
    return InfoSQLDatabaseTool(db=db).invoke(tables)


@tool("tool_query")
def tool_query(sql: str) -> str:
    """Execute a SQL query and return the result"""
    return QuerySQLDatabaseTool(db=db).invoke(sql)


@tool("tool_check")
def tool_check(sql: str) -> str:
    """
    Before executing a query, always use this tool to review the SQL query
    and correct the code if necessary.
    """
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql})


# print(tool_check.run(f"SELECT * FROM {tool_tables.run()} LIMIT 3 WHRE id=5").split('\n')[0])

print('--- Get tables: ---')
print(tool_tables.run())

print('--- Get schema: ---')
print(tool_schema.run(tool_tables.run()))

print('--- SQL Query: ---')
print(tool_query.run(f'SELECT * FROM {tool_tables.run()} LIMIT 3'))

prompt = '''Extract data with SQL query to answer {user_input}. 
   IMPORTANT: ALWAYS use the `tool_query` BEFORE returning <final answer>. 
   ONLY return <final answer> returned by `tool_query`.
   '''

# DB Agent
agent_sql = crewai.Agent(
    role='Database Engineer',
    goal=prompt,
    backstory='''
       You are an experienced database engineer that creates and optimize efficient SQL queries. 
       Use the `tool_tables` to find tables.
       Use the `tool_schema` to get the metadata for the tables.
       Use the `tool_check` to review your queries before executing.
       Use the `tool_query` to execute SQL queries and return the result.
    ''', tools=[tool_tables, tool_schema, tool_check, tool_query],
    max_iter=10,
    llm=crewai_llm,
    allow_delegation=False,
    verbose=True)

## Task
task_sql = crewai.Task(
    description=prompt,
    agent=agent_sql,
    expected_output='''Output of the query'''
)

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

# Python Data Analyst Agent & Task
prompt = '''You analyze the data received from the database engineer to answer {user_input}'''
# crewai_llm = LLM(
#     model="ollama/deepseek-coder-v2"
# )
## Agent
agent_py = crewai.Agent(
    role="Data Analyst",
    goal=prompt,
    backstory='''
        You are an experienced data anlyst that analyzes datasets using Python.
        You have attention to detail and always produce very clear and detailed results.
        First generate Python code you need to analyze the data.
        Then use the `tool_eval` to check your code.
        Finally use `tool_pycode` to execute the code and return the output.
    ''',
    tools=[tool_eval, tool_pycode],
    max_iter=10,
    llm=crewai_llm,
    allow_delegation=False, verbose=True)

## Task
task_py = crewai.Task(
    description=prompt,
    agent=agent_py,
    context=[task_sql],
    expected_output='''Output of Python code''')
#%% md
##### Test single Agent



if __name__ == '__main__':
    print(f'How many passengers in Titanic: {len(dtf)}')
    print(f'How many passengers survived: {len(dtf[dtf["Survived"] == 1])}')
    print(f'How many passengers did not survive: {len(dtf[dtf["Survived"] == 0])}')

    result = tool_pycode.run("import numpy as np; print(np.sum([1,2]))")
    print(f'\n{result}')

    print(tool_eval.run("print(Res:')"))

    # Calling data-scinetist agent
    os.environ['LITELLM_LOG'] = 'DEBUG'  # litellm.set_verbose=True
    crew = crewai.Crew(agents=[agent_py], tasks=[task_py], verbose=True)
    res = crew.kickoff(inputs={"user_input": f"how many people died in this dataset? {dtf.to_string()}"})