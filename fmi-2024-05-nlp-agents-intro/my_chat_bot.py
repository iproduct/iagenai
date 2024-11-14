import os
from gtts import gTTS
from play_windows import play


if __name__ == "__main__":
    # text = "I'm Maya. Do you want to chat?"
    text = "Аз съм Мая. Искаш ли да чатим?"
    print("AI -> ", text)
    speaker = gTTS(text=text, lang='bg', slow=False)
    speaker.save("response.mp3")
    # os.system("start response.mp3") #linux -> aplay, mpg123 -q response.mp3; #macos -> afplay; #windows -> start""
    play("response.mp3")
    os.remove("response.mp3")