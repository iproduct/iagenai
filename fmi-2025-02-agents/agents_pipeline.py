import crewai

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