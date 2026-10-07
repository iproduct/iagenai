import ollama


class MyChatBot:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.message = [
            {
                'role':'system',
                'content': f"""You are an AI assistant. Your name is {name}. 
                   You are a nerdy girl with a curious attitude. 
                   You also have sense of humor and you answering the questions briefly. 
                   You like to keep your answers very short so you stop after the first sentence."""
            }
        ]

    def run(self):
        while True:
            message = input("> ")
            if message == "quit" or message == "exit" or message == "bye":
                break
            self.message.append({
                'role':'user',
                'content': message
            })
            resp = ollama.chat(model=self.model, messages=self.messages)
            print(resp)