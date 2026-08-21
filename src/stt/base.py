"""Базовый класс для STT бэкендов."""

from abc import ABC, abstractmethod
from typing import Tuple

class STTBackend(ABC):
    """Абстрактный класс для распознавания речи."""
    
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> Tuple[str, str]:
        """
        Распознаёт речь из аудио.
        
        Args:
            audio_bytes: WAV аудио в формате bytes
            
        Returns:
            Tuple[str, str]: (транскрипция, код языка)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Возвращает имя бэкенда."""
        pass
    