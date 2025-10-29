import os

import crewai
from crewai import LLM, Agent, Task, Crew, Process

from agents_python import data_analyst, task_py

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