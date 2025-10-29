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

from agents_sql import  agent_sql, task_sql
from agents_python import data_analyst, task_py
from agents_html import agent_html, task_html


if __name__ == '__main__':
    crew = crewai.Crew(agents=[agent_sql, data_analyst, agent_html],
                       tasks=[task_sql, task_py, task_html],
                       process=crewai.Process.sequential,
                       verbose=True)

    res = crew.kickoff(inputs={"user_input": "how many people died?"})

    print("\n---Res---\n", res)