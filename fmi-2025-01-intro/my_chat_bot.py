import ollama
from gtts import gTTS
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr

class ChatBot:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.recognizer = sr.Recognizer()
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

    def is_quit_command(self, text):
        lst = ["quit", "finish", "exit", "bye"]
        return True if any(i in text.lower() for i in lst) else False

    def speech_to_text(self):
        print('Listening ...')
        with sr.Microphone() as mic:
            self.recognizer.adjust_for_ambient_noise(mic)
            audio = self.recognizer.listen(mic)
            try:
                text = self.recognizer.recognize_google(audio, language='en')
                return text
            except Exception as ex:
                print("Recognition error: ", ex)

    def text_to_speech(self, message):
        audio_obj = gTTS(text = message, lang='en', slow=False)
        audio_obj.save(self.name + '.mp3')
        audio_arr, sample_rate = sf.read(self.name + '.mp3')
        sd.play(audio_arr, sample_rate)
        sd.wait()


    def run(self):
        while True:
            text = self.speech_to_text()# input('>')
            if text is None:
                continue
            if self.is_quit_command(text):
                break
            print(f'U: {text}')
            self.messages.append({'role': 'user', 'content': text})
            resp = ollama.chat(model = self.model, messages = self.messages)
            resp_text = resp['message']['content']
            print(f'AI: {resp_text}')
            self.text_to_speech(resp_text)
            self.messages.append({'role': 'system', 'content': resp_text})

if __name__ == '__main__':
    #ChatBot demo
    maya = ChatBot('Maya', model = 'llama3.2:latest')
    maya.run()