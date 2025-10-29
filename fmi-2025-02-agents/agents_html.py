import os

import crewai
import litellm
import pandas as pd
from crewai import LLM, Agent, Task, Crew, Process
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import agent, task, CrewBase
from crewai.tools import tool, BaseTool
from langchain_experimental.utilities import PythonREPL
from crewai_tools import DirectoryReadTool, FileReadTool
from agents_python import data_analyst, task_py
from agents_sql import task_sql

# create sqlite db from Titanic dataset

# llm = LLM(model="ollama/mannix/deepseek-coder-v2-lite-instruct:latest")
# llm = LLM(model="ollama/gemma3:4b")
# llm = LLM(model="ollama/codestral:latest")
llm = LLM(model="ollama/qwen3:8b")
# llm = LLM(model="ollama/chevalblanc/gpt-4o-mini")

# HTML Developer agent
prompt = '''You write executive summary reports based on the work of the data analyst to answer {user_input}'''

## Agent
agent_html = crewai.Agent(
    role="Web Developer",
    goal=prompt,
    backstory='''
        You are an experienced web developer that writes beautiful reports using HTML and CSS.
        You always summarize texts into bullet points containing the answer to the question and noting more.
        At the end add an interactive button with JavaScript so the user can approve the report,
        and if the user clicks the button, show a pop-up text.
     ''',
    #tools=[],
    max_iter=10,
    llm=llm,
    allow_delegation=False, verbose=True)

## Task
task_html = crewai.Task(
    description=prompt,
    agent=agent_html,
    context=[task_py],
    expected_output='''HTML code''')

if __name__ == '__main__':
    crew = crewai.Crew(agents=[agent_html], tasks=[task_html], verbose=True)
    res = crew.kickoff(inputs={"user_input": "Number of people survived: 342"})
    print("\n---Res---\n", res)