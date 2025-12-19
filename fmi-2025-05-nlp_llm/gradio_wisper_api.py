import numpy as np
from gradio_client import Client, handle_file
from scipy.io.wavfile import read
import whisper

if __name__ == "__main__":
    # client = Client("http://localhost:8501/v1/models/nlp:predict")

    # Connect to a public Whisper Space (e.g., abidlabs/whisper)
    # client = Client("abidlabs/whisper")

    # Load the model (tiny, base, small, medium, large)
    model = whisper.load_model("base")

    def transcribe_audio(audio_file):
        # The audio_file parameter is a path to a temporary file
        result = model.transcribe(audio_file)
        return result["text"]

    # Use handle_file to prepare your local audio file for the API
    # audio_input = handle_file("data/bg_audio.wav")

    a = read("data/audio.wav")
    audio = np.array(a[1], dtype=float)

    result = transcribe_audio(audio)

    # Client.view_api()
    # audio_input = handle_file("data/bg_audio.wav")
    # # Predict using a local audio file
    # result = client.predict(
    #     audio=audio_input,
    #     api_name="/predict"
    # )

    print(f"Transcription: {result}")