import pyttsx3
import whisper
import torch
import string
import numpy as np
import pyaudio

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

synthesizer = pyttsx3.init()

model = whisper.load_model("medium").to(device)


def normalize_text(text):
    text = text.lower()
    text = text.replace('ё', 'е')
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ' '.join(text.split())
    return text


def respond(text):
    if "привет я разработчик" in normalize_text(text):
        response = "Сегодня выходной."
    elif "я сегодня не приду домой" in normalize_text(text):
        response = "Ну и катись отсюда."
    else:
        response = "Неизвестная команда."
    
    print("Ответ: ", response)
    synthesizer.say(response)
    synthesizer.runAndWait()

def record_audio(duration, rate, channels, chunk):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    print("Скажите что-нибудь...")

    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Запись закончена.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)


def numpy_to_torch(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
    audio_tensor = torch.from_numpy(audio_np)
    return audio_tensor


audio_bytes = record_audio(duration=5, rate=16000, channels=1, chunk=1024)
audio = numpy_to_torch(audio_bytes)

print("Распознавание...")

text = model.transcribe(audio, language='ru')["text"]

print(f"Вы сказали: {text}")

respond(text)
