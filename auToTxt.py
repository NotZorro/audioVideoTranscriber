import warnings
import os
import time
from pathlib import Path
from pydub import AudioSegment
from faster_whisper import WhisperModel
import torch
import huggingface_hub

# Игнорируем предупреждения для чистоты вывода
warnings.filterwarnings("ignore")

# Явно указываем путь к FFmpeg
AudioSegment.converter = "C:/Users/yasia/AppData/Local/ffmpeg-2025-06-02-git-688f3944ce-essentials_build/bin/ffmpeg.exe"

# Длительность сегмента в секундах (10 минут)
SEGMENT_LENGTH_S = 600

def load_audio_model(model_size="systran/faster-whisper-small"):
    print(f"Загрузка модели {model_size} для русского языка...")
    try:
        # Проверяем, включен ли оффлайн-режим
        if os.getenv("HF_HUB_OFFLINE") == "1":
            print("Оффлайн-режим (HF_HUB_OFFLINE=1) включен. Убедитесь, что модель уже загружена в кэш.")
        model = WhisperModel(
            model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16",
            local_files_only=False  # Разрешаем загрузку модели из Hugging Face Hub
        )
        print(f"Модель загружена на устройство: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        return model
    except huggingface_hub.utils._errors.LocalEntryNotFoundError as e:
        print(f"Ошибка: Модель {model_size} не найдена в локальном кэше, и доступ к интернету заблокирован. {e}")
        print("Решение: Убедитесь, что интернет доступен, и установите 'local_files_only=False'.")
        print("Также проверьте, что модель доступна на Hugging Face Hub или уже загружена в кэш (~/.cache/huggingface/hub).")
        raise
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        raise

def extract_audio_from_video(video_path):
    """Извлекает аудио из видеофайла и сохраняет его в папку audio"""
    print(f"Извлечение аудио из видео: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видеофайл {video_path} не найден")

    try:
        video = AudioSegment.from_file(video_path)
        audio_path = Path("audio") / f"{Path(video_path).stem}.wav"
        audio_path.parent.mkdir(exist_ok=True)
        video.export(audio_path, format="wav")
        print(f"Аудио сохранено в: {audio_path}")
        return audio_path
    except Exception as e:
        raise Exception(f"Ошибка при извлечении аудио: {e}. Убедитесь, что FFmpeg установлен и путь корректен.")

def split_audio(audio_path, segment_length_s=SEGMENT_LENGTH_S):
    print(f"Проверка аудиофайла: {audio_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Аудиофайл {audio_path} не найден")

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        raise Exception(f"Ошибка при загрузке аудио: {e}. Убедитесь, что FFmpeg установлен и путь корректен.")

    duration_ms = len(audio)
    segment_length_ms = segment_length_s * 1000
    segments = []
    temp_dir = Path("temp_segments")
    temp_dir.mkdir(exist_ok=True)

    print(f"Разбиваем аудио на сегменты (длительность файла: {duration_ms / 60000:.2f} минут)...")
    for i in range(0, duration_ms, segment_length_ms):
        segment = audio[i:i + segment_length_ms]
        segment_path = temp_dir / f"segment_{i // segment_length_ms}.wav"
        segment.export(segment_path, format="wav")
        segments.append(segment_path)
    return segments, temp_dir

def format_timestamp(seconds):
    """Форматирует время в секундах в формат ЧЧ:ММ:СС"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def transcribe_audio_segments(audio_path, model, language="ru", segment_length_s=SEGMENT_LENGTH_S):
    print(f"Проверка пути: {audio_path}")
    print(f"Файл существует: {os.path.exists(audio_path)}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Файл {audio_path} не найден")

    segments, temp_dir = split_audio(audio_path, segment_length_s)
    transcription = []
    current_speaker = 1
    last_end_time = 0.0
    segment_length_ms = segment_length_s * 1000

    print(f"Обработка {len(segments)} сегментов...")
    for i, segment_path in enumerate(segments):
        print(f"Расшифровка сегмента {i + 1}/{len(segments)}: {segment_path}")
        try:
            start_time = time.time()
            segments_transcribed, _ = model.transcribe(
                str(segment_path),
                language=language,
                beam_size=5,
                vad_filter=True,
               # length_penalty=1,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            segment_offset = i * segment_length_s
            for seg in segments_transcribed:
                start_time_seg = seg.start + segment_offset
                end_time_seg = seg.end + segment_offset
                text = seg.text.strip()

                # Считаем паузу > 1 секунды как новую реплику
                if start_time_seg - last_end_time > 1.0 and text:
                    current_speaker += 1
                last_end_time = end_time_seg

                if text:  # Пропускаем пустые сегменты
                    transcription.append(
                        f"Реплика {current_speaker} [{format_timestamp(start_time_seg)} - {format_timestamp(end_time_seg)}]: {text}")
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Время обработки сегмента {i + 1}: {processing_time:.2f} секунд")
        except Exception as e:
            print(f"Ошибка при обработке сегмента {segment_path}: {e}")
        finally:
            if os.path.exists(segment_path):
                os.remove(segment_path)

    if temp_dir.exists():
        temp_dir.rmdir()

    return "\n".join(transcription)

def main():
    # Запрашиваем путь к файлу
    input_path = input("Введите путь к аудио- или видеофайлу (например, audio/1.ogg или audio/1.mp4): ").strip()

    # Проверяем существование файла
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл {input_path} не найден")
        return

    # Создаем папку text, если не существует
    output_dir = Path("texts")
    output_dir.mkdir(exist_ok=True)

    model_size = "systran/faster-whisper-small"  # Модель для русского языка
    try:
        model = load_audio_model(model_size)
    except Exception as e:
        print(f"Не удалось загрузить модель: {e}")
        print("Проверьте интернет-соединение, настройки кэша и убедитесь, что модель доступна на Hugging Face.")
        return

    try:
        # Проверяем, является ли файл видео
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv"}
        is_video = Path(input_path).suffix.lower() in video_extensions

        if is_video:
            audio_path = extract_audio_from_video(input_path)
        else:
            audio_path = input_path

        # Транскрипция
        transcription = transcribe_audio_segments(audio_path, model, language="ru")
        print("\nРезультат транскрипции:")
        print(transcription)

        # Сохраняем результат в папку texts с учетом расширения исходного файла
        output_path = output_dir / f"{Path(input_path).stem}_{Path(input_path).suffix[1:]}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"\nТранскрипция сохранена в {output_path}")
        return
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()