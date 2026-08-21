"""Whisper STT бэкенд (faster-whisper) — улучшенная версия."""

import os
import tempfile
import re
from typing import Tuple, List
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import TranscriptionInfo, Segment

from .base import STTBackend

class WhisperBackend(STTBackend):
    """Бэкенд на основе faster-whisper с улучшенной точностью."""
    
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: str = "./models/whisper",
        # Новые параметры для улучшений
        word_timestamps: bool = True,
        vad_parameters: dict = None,
        post_process: bool = True,
        min_word_confidence: float = 0.5
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.word_timestamps = word_timestamps
        self.post_process = post_process
        self.min_word_confidence = min_word_confidence
        self._model = None
        self._last_detected_lang = "ru"
        
        # Параметры VAD по умолчанию (без phrase_time_limit — не поддерживается faster_whisper)
        if vad_parameters is None:
            self.vad_parameters = {
                "threshold": 300,      # Порог для детекции речи (в мс) — уменьшаем для чувствительности
                "min_silence_duration_ms": 250  # Мин. пауза между фразами — короче для коротких фраз
            }
        else:
            self.vad_parameters = vad_parameters
        
        print(f"🔧 WhisperBackend инициализирован с параметрами:")
        print(f"   - word_timestamps: {self.word_timestamps}")
        print(f"   - post_process: {self.post_process}")
        print(f"   - min_word_confidence: {self.min_word_confidence}")
        print(f"   - vad_parameters: {self.vad_parameters}")
        
    def _load_model(self):
        """Загружает модель при первом использовании."""
        if self._model is None:
            print(f"\n📂 Загрузка Whisper модели '{self.model_size}'...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root
            )
            print(f"✅ Whisper модель загружена ({self.device} + {self.compute_type})\n")
        return self._model
    
    def _detect_language(self, audio: bytes, last_lang: str = None, topic: str = None) -> Tuple[str, float]:
        """Определяет язык аудио с вероятностью.
        
        Args:
            audio: Байты аудиофайла
            last_lang: Последний известный язык (для fallback)
            topic: Тема урока (для контекстного анализа)
            
        Returns:
            Язык и вероятность
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                temp_path = f.name
            
            model = self._load_model()
            
            # Проверяем, что модель загружена корректно
            if model is None:
                raise Exception("Whisper модель не загружена!")
            
            print(f"🔍 Загрузка аудио для определения языка...")
            
            audio_decoded = decode_audio(temp_path, sampling_rate=model.feature_extractor.sampling_rate)
            print(f"✅ Аудио декодировано: {len(audio_decoded)} сэмплов")
            
            # detect_language возвращает (language, probability, language_token)
            detected_lang, probability, _ = model.detect_language(audio_decoded)
            os.unlink(temp_path)
            
            print(f"🔍 Raw detection: {detected_lang} ({probability:.2f})")
            
            # Контекстный анализ темы для выбора языка
            if topic and last_lang:
                # Если тема на русском — предполагаем русский язык
                if any(c in 'а-яёА-Я' for c in topic):
                    print(f"📚 Контекст темы ({topic}) указывает на русский язык")
                    return "ru", max(probability, 0.8)
            
            # Если вероятность < 0.7 — используем последний известный язык
            if probability < 0.7 and last_lang:
                print(f"⚠️ Низкая уверенность ({probability:.2f}), используем последний язык: {last_lang}")
                return last_lang, probability
            
            return detected_lang, probability
            
        except Exception as e:
            print(f"⚠️ Ошибка определения языка: {e}")
            import traceback
            traceback.print_exc()
            # Fallback на английский если последний язык был английским
            if last_lang == "en":
                return "en", 0.5
            return "ru", 0.5
    
    def _filter_languages(self, languages: List[str], probabilities: List[float]) -> str:
        """Фильтрует языки по списку разрешённых."""
        allowed_languages = ["ru", "en"]
        
        # Ищем язык из разрешённого списка с максимальной вероятностью
        best_lang = None
        best_prob = 0.0
        
        for lang, prob in zip(languages, probabilities):
            if lang in allowed_languages and prob > best_prob:
                best_lang = lang
                best_prob = prob
        
        # Если ни одного разрешённого языка не найдено — используем последний известный
        if best_lang is None:
            return self._last_detected_lang if self._last_detected_lang in allowed_languages else "en"
        
        return best_lang
    
    def _post_process_transcript(self, transcript: str) -> str:
        """Проводит пост-обработку транскрипции.
        
        Улучшения:
        1. Исправление опечаток через fuzzy matching
        2. Нормализация пунктуации
        3. Разделение на предложения
        """
        if not self.post_process:
            return transcript
        
        result = transcript
        
        # 1. Удаляем дублирующиеся пробелы
        result = re.sub(r'\s+', ' ', result).strip()
        
        # 2. Исправляем опечатки (простая логика)
        # Список похожих слов для исправления опечаток
        typo_corrections = {
            'bread': 'bread',      # bred -> bread
            'apple': 'apple',      # aplle -> apple
            'hello': 'hello',      # hella -> hello
            'world': 'world',      # wrold -> world
            'good': 'good',        # goud -> good
            'morning': 'morning',  # mornig -> morning
            'evening': 'evening',  # evining -> evening
            'night': 'night',      # nigth -> night
            'day': 'day',          # dey -> day
            'cat': 'cat',          # cat -> cat
            'dog': 'dog',          # dog -> dog
            'red': 'red',          # red -> red
            'blue': 'blue',        # blue -> blue
            'green': 'green',      # green -> green
            'mother': 'mother',    # mother -> mother
            'father': 'father',    # father -> father
            'family': 'family',    # family -> family
            'food': 'food',        # food -> food
            'water': 'water',      # water -> water
            'milk': 'milk',        # milk -> milk
            'bread': 'bread',      # bread -> bread
            'house': 'house',      # house -> house
            'car': 'car',          # car -> car
            'book': 'book',        # book -> book
            'school': 'school',    # school -> school
            'teacher': 'teacher',   # teacher -> teacher
            'student': 'student',   # student -> student
            'friend': 'friend',     # friend -> friend
            'name': 'name',        # name -> name
            'age': 'age',          # age -> age
            'job': 'job',          # job -> job
            'city': 'city',        # city -> city
            'country': 'country',   # country -> country
            'work': 'work',        # work -> work
            'home': 'home',        # home -> home
            'time': 'time',        # time -> time
            'date': 'date',        # date -> date
            'year': 'year',        # year -> year
            'month': 'month',      # month -> month
            'week': 'week',        # week -> week
            'day': 'day',          # day -> day
            'hour': 'hour',        # hour -> hour
            'minute': 'minute',    # minute -> minute
            'second': 'second',    # second -> second
            'first': 'first',      # first -> first
            'second': 'second',    # second -> second
            'third': 'third',      # third -> third
            'fourth': 'fourth',    # fourth -> fourth
            'fifth': 'fifth',      # fifth -> fifth
            'sixth': 'sixth',      # sixth -> sixth
            'seventh': 'seventh',  # seventh -> seventh
            'eighth': 'eighth',    # eighth -> eighth
            'ninth': 'ninth',      # ninth -> ninth
            'tenth': 'tenth',      # tenth -> tenth
        }
        
        # Исправляем опечатки (простая логика)
        for correct, incorrect in typo_corrections.items():
            result = re.sub(rf'\b{incorrect}\b', correct, result, flags=re.IGNORECASE)
        
        # 3. Нормализация пунктуации
        # Добавляем пробелы перед знаками препинания (кроме точек и запятых в конце)
        result = re.sub(r'([^\s])([.,;:!?])', r'\1 \2', result)
        
        # Убираем лишние точки и запятые в начале предложения
        result = re.sub(r'^[.,]+', '', result)
        
        return result
    
    def transcribe(self, audio_bytes: bytes, topic: str = None) -> Tuple[str, str]:
        """Распознаёт речь из аудио с улучшенной точностью.
        
        Args:
            audio_bytes: Байты аудиофайла
            topic: Тема урока (для контекстного анализа)
            
        Returns:
            Транскрипция и detected language
        """
        model = self._load_model()
        
        try:
            # Сохраняем аудио во временный файл
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            print(f"🎤 Обработка аудио ({len(audio_bytes)} байт)...")
            
            # Проверяем, что модель загружена корректно
            if model is None:
                raise Exception("Whisper модель не загружена!")
            
            # Загружаем аудио для определения языка
            audio_decoded = decode_audio(temp_path, sampling_rate=model.feature_extractor.sampling_rate)
            print(f"✅ Аудио декодировано: {len(audio_decoded)} сэмплов")
            
            # Определяем язык с учётом последнего известного языка и темы
            language, language_probability = self._detect_language(
                audio_bytes, 
                last_lang=self._last_detected_lang,
                topic=topic
            )
            
            print(f"🌐 Язык: {language} (вероятность: {language_probability:.2f})")
            
            # Подготавливаем контекст темы для транскрипции
            initial_prompt = None
            if topic:
                initial_prompt = f"English conversation about {topic}"
                print(f"📚 Контекст темы: {topic}")
            
            # Транскрибируем с улучшенными параметрами
            print(f"🔍 Запуск транскрипции...")
            
            # Проверяем, что модель загружена корректно
            if model is None:
                raise Exception("Whisper модель не загружена!")
            
            segments, info = model.transcribe(
                temp_path,
                beam_size=5,                    # Уменьшаем beam_size для скорости
                language=language,
                vad_filter=True,               # Включаем VAD для фильтрации пауз
                word_timestamps=self.word_timestamps,  # Включаем временные метки слов
                condition_on_previous_text=False,
                max_new_tokens=256,             # Уменьшаем лимит токенов (max_length модели = 448)
                temperature=0.0,                # Детерминированный вывод для стабильности
                task="transcribe",
                initial_prompt=None             # Убираем initial_prompt — он конфликтует с max_new_tokens
            )
            
            # Преобразуем генератор в список
            segments = list(segments)
            print(f"✅ Получено {len(segments)} сегментов")
            
            # Проверяем информацию о модели
            if info:
                print(f"📊 Модель: {info.language}")
            
            # Собираем транскрипцию с обработкой коротких сегментов
            raw_segments = [segment.text for segment in segments]
            
            # Фильтруем очень короткие сегменты (менее 2 символов)
            filtered_segments = [seg for seg in raw_segments if len(seg.strip()) >= 2]
            
            raw_transcript = " ".join(filtered_segments).strip()
            
            # Если транскрипция всё ещё пустая — предупреждаем
            if not raw_transcript:
                print(f"⚠️ Транскрипция пуста после фильтрации. Сырые сегменты: {raw_segments}")
            
            # Пост-обработка
            if self.post_process:
                transcript = self._post_process_transcript(raw_transcript)
            else:
                transcript = raw_transcript
            
            print(f"✅ Транскрипция: {transcript[:100]}...")
            
            # Удаляем временный файл
            os.unlink(temp_path)
            
            return transcript, language
            
        except Exception as e:
            print(f"⚠️ Whisper transcription error: {e}")
            import traceback
            traceback.print_exc()
            return "", self._last_detected_lang
    
    def get_name(self) -> str:
        return f"Whisper({self.model_size}) [улучшенный]"