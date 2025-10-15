import pandas as pd
from langchain_ollama import OllamaLLM
import sqlite3
from langchain_community.tools import ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDatabaseTool

from pandas.io.sql import SQLDatabase

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

    db = SQLDatabase('sqlite:///titanic.db')



