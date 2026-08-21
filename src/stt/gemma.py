"""Gemma 4 STT бэкенд (мультимодальный)."""

import os
import tempfile
import base64
from typing import Tuple

import httpx

from .base import STTBackend

class GemmaBackend(STTBackend):
    """Бэкенд на основе Gemma 4 (с аудиовходом)."""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma4:e2b",
        language: str = "ru"
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.language = language
        self._last_detected_lang = language
        
    def transcribe(self, audio_bytes: bytes) -> Tuple[str, str]:
        """Распознаёт речь через Gemma 4."""
        try:
            # Кодируем аудио в base64
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # Формируем запрос к Gemma
            # Примечание: API может отличаться в зависимости от версии
            prompt = (
                f"Распознай речь на языке '{self.language}'. "
                f"Верни только транскрипцию, без пояснений. "
                f"Аудио: {audio_base64}"
            )
            
            # Здесь нужна адаптация под API Gemma 4
            # В текущей версии Ollama может не поддерживать аудиовход
            # Это заглушка для будущей реализации
            
            # with httpx.Client(timeout=30.0) as client:
            #     response = client.post(
            #         f"{self.ollama_url}/api/generate",
            #         json={
            #             "model": self.model,
            #             "prompt": prompt,
            #             "stream": False,
            #             "options": {
            #                 "temperature": 0.0,
            #                 "num_predict": 256
            #             }
            #         }
            #     )
            #     if response.status_code == 200:
            #         result = response.json()
            #         transcript = result.get("response", "").strip()
            #         return transcript, self.language
            
            # Пока возвращаем заглушку
            print(f"⚠️ Gemma STT ещё не реализован, используем fallback")
            return "", self.language
            
        except Exception as e:
            print(f"⚠️ Gemma transcription error: {e}")
            return "", self._last_detected_lang
    
    def get_name(self) -> str:
        return f"Gemma({self.model})"
    