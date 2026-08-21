"""STT бэкенды для Parlor Vision."""

import os
from typing import Optional, Dict, Any

from .base import STTBackend
from .whisper import WhisperBackend
from .gemma import GemmaBackend

__all__ = ["STTBackend", "WhisperBackend", "GemmaBackend", "create_stt_backend"]

def create_stt_backend(
    backend_type: Optional[str] = None,
    model_size: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
    download_root: str = "./models/whisper",
    word_timestamps: bool = True,
    vad_parameters: Dict[str, Any] = None,
    post_process: bool = True
) -> STTBackend:
    """
    Создаёт STT бэкенд на основе конфигурации.
    
    Args:
        backend_type: "whisper" или "gemma". Если None — берётся из .env.
        model_size: Размер модели Whisper (large-v3-turbo, base-300M, small-300M).
        device: Устройство для inference (cuda, cpu, auto).
        compute_type: Тип вычислений (float16, int8, float32).
        download_root: Папка для загрузки моделей.
        word_timestamps: Включать временные метки для каждого слова.
        vad_parameters: Параметры Voice Activity Detection.
        post_process: Включать пост-обработку транскрипции.
        
    Returns:
        STTBackend: экземпляр бэкенда
    """
    if backend_type is None:
        backend_type = os.environ.get("STT_BACKEND", "whisper").lower()
    
    if backend_type == "gemma":
        return GemmaBackend(
            ollama_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("OLLAMA_MODEL", "gemma4:e2b"),
            language=os.environ.get("STT_LANGUAGE", "ru")
        )
    else:
        # Whisper по умолчанию
        return WhisperBackend(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            word_timestamps=word_timestamps,
            vad_parameters=vad_parameters,
            post_process=post_process
        )


