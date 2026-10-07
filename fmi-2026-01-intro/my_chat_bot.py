import ollama
from gtts import gTTS
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr


class ChatBot:
    def __init__(self, name, model):
        self.name = name
        self.model = model


class MyChatBot:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.recognizer = sr.Recognizer()
        self.messages = [
            {
                'role':'system',
                'content': f"""You are an AI assistant. Your name is {name}. 
                   You are a nerdy girl with a curious attitude. 
                   You also have sense of humor and you answering the questions briefly. 
                   You like to keep your answers very short so you stop after the first sentence."""
            }
        ]

    def is_quit_command(self, message):
        lst = ['quit', 'exit', 'stop', 'finish', 'bye', 'бай-бай']
        return True if any(sw in message.lower() for sw in lst) else False

    def text_to_speech(self, text):
        audio = gTTS(text=text, lang='en', slow=False)
        audio.save(f'{self.name}.mp3')
        audio_arr, sample_rate = sf.read(f'{self.name}.mp3')
        sd.play(audio_arr, sample_rate)
        sd.wait()

    def speech_to_text(self):
        print(f'{self.name} is listening ...')
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio, language='en')
                return text
            except sr.UnknownValueError as ex:
                ...
            except Exception as ex:
                print(ex)

    def run(self):
        while True:
            message = input("> ")
            if self.is_quit_command(message):
                break
            self.messages.append({
                'role':'user',
                'content': message
            })
            resp = ollama.chat(model=self.model, messages=self.messages)
            print(resp.message.content)

if __name__ == '__main__':
    bot = MyChatBot(name="Maya", model="llama3.2:latest")
    bot.run()