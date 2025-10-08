import ollama


class ChatBot:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.messages = [
            {
                'role': 'system',
                'content': f""" You are an AI assistant. Your name is {self.name}.
                You spoiled russian girl with a nasty attiytude and strong accent.
                You also have dry humor and you always make fun of everyone when answering the question.
                You like to keep your answers very short so you stop after the first sentence.
                """
            }
        ]

    def run(self):
        while True:
            text = input('>')
            if text == 'quit':
                break
            print(f'U: {text}')
            self.messages.append({'role': 'user', 'content': text})
            resp = ollama.chat(model = self.model, messages = self.messages)
            resp_text = resp['message']['content']
            print(f'AI: {resp_text}')
            self.messages.append({'role': 'system', 'content': resp_text})

if __name__ == '__main__':
    #ChatBot demo
    maya = ChatBot('Maya', model = 'llama3.2:latest')
    maya.run()