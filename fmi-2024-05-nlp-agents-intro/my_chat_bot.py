import os

import ollama
from gtts import gTTS
import speech_recognition as sr
from play_windows import play

class ChatBot():
    def __init__(self, name, model):
        print("--- Starting up ", name, "---")
        self.name = name
        self.model = model
        self.recognizer = sr.Recognizer()


    # Speech recognition using SpeechRecognition
    def speech_to_text(self):
        print("Listening ...")
        with sr.Microphone() as mic:
            self.recognizer.adjust_for_ambient_noise(mic, duration=1)
            audio = self.recognizer.listen(mic)
        try:
            text = self.recognizer.recognize_google(audio)
            print("You -> ", text)
            return text
        except Exception as e:
            print("You -> Recognition Error", e)
            return None

    # Speech generation using gTTS
    def text_to_speech(self, text):
        speaker = gTTS(text=text, lang='en', slow=False)
        speaker.save("response.mp3")
        # os.system("start response.mp3") #linux -> aplay, mpg123 -q response.mp3; #macos -> afplay; #windows -> start""
        play("response.mp3")
        os.remove("response.mp3")

    def what(self, text):
        lst = ["what are you", "who are you"]
        return True if any(i in text.lower() for i in lst) else False

    def run(self):
        text= ''
        while text is None or len(text.strip()) == 0 or ('quit' not in text.strip().lower()):
            text = self.speech_to_text()



            if text is not None:
                ## what
                if self.what(text) is True:
                    first_sentence = "I am an AI created by Trayan"
                else:
                    resp = ollama.generate(model=self.model, prompt=text)['response']
                    first_sentence = resp.split('\n')[0]
                print("AI -> ", first_sentence)
                maya.text_to_speech(first_sentence)
        print(text)


if __name__ == "__main__":

    # ChatBot demo
    maya = ChatBot('Maya', 'maya')
    maya.run()
    # text = maya.speech_to_text()
    # if text is not None:
    #     maya.text_to_speech(text)

    # Speech generation using gTTS
    # resp = ollama.chat(model='llama3.1', messages=[
    #     {
    #         'role': 'system',
    #         'content': 'You are an AI assistant.',
    #     },
    #     {
    #         'role': 'user',
    #         'content': 'Why is the sky blue?',
    #     },
    # ])
    # text = resp['message']['content']
    # print(text)
    # # text = "I'm Maya. Do you want to chat?"
    # # text = "Аз съм Мая. Искаш ли да чатим?"
    # print("AI -> ", text)
    # speaker = gTTS(text=text, lang='en', slow=False)
    # speaker.save("response.mp3")
    # # os.system("start response.mp3") #linux -> aplay, mpg123 -q response.mp3; #macos -> afplay; #windows -> start""
    # play("response.mp3")
    # os.remove("response.mp3")

    # Speech recognition using SpeechRecognition
    # recognizer = sr.Recognizer()
    # with sr.Microphone() as source:
    #     recognizer.adjust_for_ambient_noise(source, duration=1)
    #     print("Listening ...")
    #     audio = recognizer.listen(source)
    # try:
    #    text = recognizer.recognize_google(audio)
    #    print("You -> ", text)
    # except Exception as e:
    #     print("You -> Recognition Error", e.message)
