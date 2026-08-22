"""Упрощённый Whisper STT бэкенд (faster-whisper)."""

import os
import tempfile
from typing import Tuple, Optional

from faster_whisper import WhisperModel

from .base import STTBackend


class WhisperBackend(STTBackend):
    """Бэкенд на основе faster-whisper."""
    
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: str = "./models/whisper",
        word_timestamps: bool = True,
        post_process: bool = True,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.word_timestamps = word_timestamps
        self.post_process = post_process
        self._model: Optional[WhisperModel] = None
        self._last_detected_lang: str = "ru"
    
    def _load_model(self) -> WhisperModel:
        """Загружает модель при первом использовании."""
        if self._model is None:
            print(f"Загрузка Whisper модели '{self.model_size}'...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root
            )
        return self._model
    
    def _detect_language(self, audio_bytes: bytes, last_lang: Optional[str] = None) -> Tuple[str, float]:
        """Определяет язык аудио.
        
        Args:
            audio_bytes: Байты аудиофайла
            last_lang: Последний известный язык (для fallback)
            
        Returns:
            Язык и вероятность
        """
        try:
            model = self._load_model()
            # Создаём временный файл для аудио
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)
            
            # Используем путь к файлу для определения языка
            segments, info = model.transcribe(
                tmp_path,
                language=None,  # Автоопределение
                beam_size=8,
                vad_filter=True,
                initial_prompt="PAUSE PAUSE",
                task="transcribe",
                max_new_tokens=10,
            )
            
            # Удаляем временный файл
            os.unlink(tmp_path)
            
            # Получаем язык из info
            detected_lang = info.language if hasattr(info, 'language') else None
            
            # Если язык не определён — используем последний известный язык
            if detected_lang is None and last_lang:
                return last_lang, 0.5
            
            return detected_lang or "ru", 0.5
            
        except Exception as e:
            print(f"Ошибка определения языка: {e}")
            if last_lang == "en":
                return "en", 0.5
            return "ru", 0.5
    
    def _post_process_transcript(self, transcript: str) -> str:
        """Проводит пост-обработку транскрипции."""
        if not self.post_process:
            return transcript
        
        import re
        result = transcript
        
        # Удаляем дублирующиеся пробелы
        result = re.sub(r'\s+', ' ', result).strip()
        
        # Нормализация пунктуации — добавляем пробелы перед знаками препинания
        result = re.sub(r'([^\s])([.,;:!?])', r'\1 \2', result)
        
        # Убираем лишние точки и запятые в начале строки
        result = re.sub(r'^[.,;:!?]+', '', result)
        
        return result
    
    def transcribe(self, audio_bytes: bytes, lang: Optional[str] = None, keywords: Optional[list[str]] = None) -> Tuple[str, str]:
        """Распознаёт речь из аудио.
        
        Args:
            audio_bytes: Байты аудиофайла
            lang: Язык транскрипции (опционально)
            keywords: Ключевые слова для initial_prompt (опционально)
            
        Returns:
            Транскрипция и detected language
        """
        model = self._load_model()
        
        try:
            print(f"Обработка аудио ({len(audio_bytes)} байт)...")
            
            if lang:
                language = lang
            else:
                # Определяем язык
                language, language_probability = self._detect_language(
                    audio_bytes,
                    last_lang=self._last_detected_lang
                )
                print(f"Язык: {language} (вероятность: {language_probability:.2f})")


            initial_prompt = "PAUSE PAUSE"
            if keywords:
                initial_prompt = ". ".join(keywords[:15]) + "."
                print(f"🔑 Язык: initial_prompt из keywords: {initial_prompt[:50]}...")

            # Создаём временный файл для аудио
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)
            
            # Транскрибируем с использованием пути к файлу
            segments, info = model.transcribe(
                tmp_path,
                beam_size=5,
                language=language,
                vad_filter=True,
                word_timestamps=self.word_timestamps,
                condition_on_previous_text=False,
                max_new_tokens=256,
                temperature=0.0,
                initial_prompt=initial_prompt,
                task="transcribe",
            )
            
            # Удаляем временный файл
            os.unlink(tmp_path)
            
            # Преобразуем генератор в список
            segments = list(segments)
            
            # Собираем транскрипцию
            raw_transcript = " ".join(segment.text for segment in segments).strip()
            
            # Пост-обработка
            transcript = self._post_process_transcript(raw_transcript)
            
            return transcript, language
            
        except Exception as e:
            print(f"Ошибка транскрипции: {e}")
            import traceback
            traceback.print_exc()
            return "", self._last_detected_lang
    
    def get_name(self) -> str:
        return f"Whisper({self.model_size})"