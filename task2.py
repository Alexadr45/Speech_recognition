from pyannote.audio import Pipeline
import librosa
import whisper
import torch


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Инициализация диаризационной модели
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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").to(torch.device(device))

# Путь к аудиофайлу
audio_file = "catch_me.wav"

# Проведение диаризации
diarization = pipeline(audio_file, num_speakers=2)

# Загрузка аудио файла и его нормализация
audio, rate = librosa.load(audio_file, sr=16000)

previous_speaker = None
audio_segments = []
# Обработка каждого сегмента
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if turn.end - turn.start < 0.1:
        continue
    if previous_speaker != speaker:
        if previous_speaker:
            audio_segments.append((speaker, audio[start:end]))
        previous_speaker = speaker
        start = int(turn.start * rate)
        end = int(turn.end * rate)
    else:
        end = int(turn.end * rate)
audio_segments.append((speaker, audio[start:end]))

# Грузим модель для распознавания
model = whisper.load_model("medium").to(device)

for speaker, audio_segment in audio_segments:
    transcription = model.transcribe(audio_segment, language='ru')["text"]
    print(f"{speaker}: {transcription}")