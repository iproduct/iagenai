import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# llm = OllamaLLM(model="codestral:latest")
llm = OllamaLLM(model="qwen3:8b")
dtf = pd.read_csv("data/data_titanic.csv")

agent = create_pandas_dataframe_agent(llm=llm, df=dtf, verbose=True, allow_dangerous_code=True,
                                      agent_executor_kwargs={'handle_parsing_errors': True}, max_iterations=100)
if __name__ == '__main__':
    agent.invoke("how many people died? Prefix python code with 'python' instead of 'py'")

    agent.invoke('''
            You are an experienced data scientist that does machine learning using Python and sckit-learn.
            Take the dataframe provided as 'df' parameter and split into train set and test set. 
            Then train a simple classification to predict the column `Survived`.
            Then use the score to evaluate the model predictions.
            Run the generated Python program and print the results to the console.
            Prefix python code with 'python' instead of 'py'.
            Print the results to the console.''')