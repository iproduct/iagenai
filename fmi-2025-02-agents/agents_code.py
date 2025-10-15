import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
import sqlite3
from langchain_community.tools import ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDatabaseTool
# from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain_community.tools.sql_database.tool import (
#     InfoSQLDatabaseTool,
#     ListSQLDatabaseTool,
#     QuerySQLCheckerTool,
#     QuerySQLDatabaseTool,
# )

# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

from langchain_community.utilities.sql_database import SQLDatabase

if __name__ == '__main__':
    dtf = pd.read_csv("data_titanic.csv")
    print(dtf.head(3))
    llm = OllamaLLM(model="qwen3:8b")

    print(f'How many passengers in Titanic: {len(dtf)}')
    print(f'How many passengers survived: {len(dtf[dtf["Survived"] == 1])}')
    print(f'How many passengers did not survive: {len(dtf[dtf["Survived"] == 0])}')

    # res = llm.invoke('''
    # Answer short with a single number:
    # Do you know 'Titanic' dataset from Kaggle - If yes, tell me how many people survived.    ''')
    # print(res)

    # create sqlite db from Titanic dataset
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
        """Execute a SQL query"""
        return QuerySQLDatabaseTool(db=db).invoke(sql)


    print('--- Get tables: ---')
    print(tool_tables.run(''))

    print('--- Get schema: ---')
    print(tool_schema.run(tool_tables.run('')))

    print('--- SQL Query: ---')
    print(tool_query.run(f'SELECT * FROM {tool_tables.run('')} LIMIT 3'))
