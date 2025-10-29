import crewai
import pandas as pd
from crewai import LLM
from crewai.tools import tool
from langchain_ollama import OllamaLLM
import sqlite3
from langchain_community.tools import ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDatabaseTool, QuerySQLCheckerTool

# from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain_community.tools.sql_database.tool import (
#     InfoSQLDatabaseTool,
#     ListSQLDatabaseTool,
#     QuerySQLCheckerTool,
#     QuerySQLDatabaseTool,
# )
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

from langchain_community.utilities.sql_database import SQLDatabase

# llm = OllamaLLM(model="qwen3:8b")

# crewai_llm = LLM(model="ollama/mannix/deepseek-coder-v2-lite-instruct:latest")
# crewai_llm = LLM(model="ollama/codestral:latest")
llm = LLM(model="ollama/qwen3:8b")



# res = llm.invoke('''
# Answer short with a single number:
# Do you know 'Titanic' dataset from Kaggle - If yes, tell me how many people survived.    ''')
# print(res)

# create sqlite db from Titanic dataset
dtf = pd.read_csv("data/data_titanic.csv")
dtf.to_sql(index = False, name='titanic', con=sqlite3.connect("titanic.db"), if_exists='replace')

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
    llm=llm,
    allow_delegation=False,
    verbose=True)

## Task
task_sql = crewai.Task(
    description=prompt,
    agent=agent_sql,
    expected_output='''Output of the query'''
)

if __name__ == '__main__':

    print(dtf.head(3))

    print(f'How many passengers in Titanic: {len(dtf)}')
    print(f'How many passengers survived: {len(dtf[dtf["Survived"] == 1])}')
    print(f'How many passengers did not survive: {len(dtf[dtf["Survived"] == 0])}')

    # print(tool_check.run(f"SELECT * FROM {tool_tables.run()} LIMIT 3 WHRE id=5").split('\n')[0])

    print('--- Get tables: ---')
    print(tool_tables.run())

    print('--- Get schema: ---')
    print(tool_schema.run(tool_tables.run()))

    print('--- SQL Query: ---')
    print(tool_query.run(f'SELECT * FROM {tool_tables.run()} LIMIT 3'))


    # Test DB Agent
    crew = crewai.Crew(agents=[agent_sql], tasks=[task_sql], verbose=False)
    res = crew.kickoff(inputs={"user_input":"how many people died? Return a plain number. Return <final answer>  ONLY IF returned by `tool_query` otherwise continue thinking."})
    # res = crew.kickoff(inputs={"user_input":"how many people survived?"})
    print("\nResponse:\n", res)