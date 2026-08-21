"""TTS multiplataforma: Kokoro-ONNX (primario), edge-tts (fallback Windows)."""

import re
import os
import sys
import numpy as np


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str = None, speed: float = 1.1, lang: str = "en") -> np.ndarray:
        raise NotImplementedError


class KokoroONNXBackend(TTSBackend):
    """kokoro-onnx backend."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        print("Cargando Kokoro-ONNX...")
        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000
        
        self.voice_map = {
            "en": "af_heart",   # American Female
            "es": "ef_dora",    # Spanish Female (Nativa)
            "ru": "af_heart"    # Russian Female (русский) - проверьте, есть ли такой голос
        }

    def generate(self, text: str, voice: str = None, speed: float = 1.1, lang: str = "en") -> np.ndarray:
        selected_voice = voice or self.voice_map.get(lang, "af_heart")
        print(f"🔊 Generando audio ({lang}) con voz: {selected_voice}")
        
        # Le pasamos explícitamente el idioma 'lang' a create() para que la 
        # fonetización (G2P) se haga con reglas de español y no de inglés.
        pcm, _sr = self._model.create(text, voice=selected_voice, speed=speed, lang=lang)
        return pcm


class EdgeTTSBackend(TTSBackend):
    """edge-tts backend (Microsoft) - CALIDAD PROFESIONAL PARA ESPAÑOL."""
    
    def __init__(self):
        self.sample_rate = 24000
        self.voice_map = {
            "en": "en-US-EmmaNeural", 
            "ru": "ru-RU-SvetlanaNeural"
        }
        
    def generate(self, text: str, voice: str = None, speed: float = 1.1, lang: str = "en") -> np.ndarray:
        import asyncio
        import io
        import re
        import edge_tts
        import soundfile as sf

        clean_text = clean_text_for_tts(text)
        has_ssml = bool(re.search(r'<lang\s+xml:lang=', clean_text))
        
        selected_voice = voice or self.voice_map.get(lang, "en-US-EmmaNeural")
        print(f"🔊 Generando audio Edge-TTS ({lang}) con voz: {selected_voice} {'(SSML)' if has_ssml else ''}")
        
        rate = f"{int((speed - 1) * 100):+d}%"
            
        async def _generate():
            communicate = edge_tts.Communicate(clean_text, selected_voice, rate=rate)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        audio_data = loop.run_until_complete(_generate())
        audio, sr = sf.read(io.BytesIO(audio_data))
        
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        return audio


def load() -> TTSBackend:
    """Load the best available TTS backend para Windows."""
    # Intentar kokoro-onnx primero (recomendado, offline)
    # try:
    #     backend = KokoroONNXBackend()
    #     print(f"TTS: kokoro-onnx (sample_rate={backend.sample_rate})")
    #     return backend
    # except Exception as e:
    #     print(f"TTS: kokoro-onnx falló ({e}), intentando edge-tts...")
        
    # Fallback a edge-tts (requiere internet)
    try:
        backend = EdgeTTSBackend()
        print(f"TTS: edge-tts (sample_rate={backend.sample_rate})")
        return backend
    except Exception as e:
        print(f"TTS: edge-tts falló ({e})")
        
    raise RuntimeError(
        "No se pudo cargar ningún backend de TTS. "
        "Instala kokoro-onnx o edge-tts: pip install kokoro-onnx o pip install edge-tts"
    )

def clean_text_for_tts(text: str) -> str:
    """Удаляет маркдаун-разметку и специальные символы из текста для TTS."""
    if not text:
        return ""
    
    # Сохраняем SSML-теги, заменяя их на временные маркеры
    ssml_tags = []
    def save_ssml(match):
        ssml_tags.append(match.group(0))
        return f"__SSML_{len(ssml_tags)-1}__"
    
    # Находим и сохраняем все теги <lang>
    text = re.sub(r'<lang\s+xml:lang="[^"]*">.*?</lang>', save_ssml, text, flags=re.DOTALL)
    
    # Удаляем маркдаун
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **жирный**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *курсив*
    text = re.sub(r'__(.*?)__', r'\1', text)      # __подчёркнутый__
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # ~~зачёркнутый~~
    text = re.sub(r'`(.*?)`', r'\1', text)        # `код`
    text = re.sub(r'#+ ', '', text)               # заголовки
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # ссылки
    
    # Возвращаем SSML-теги на место
    for i, tag in enumerate(ssml_tags):
        text = text.replace(f"__SSML_{i}__", tag)
    
    return text.strip()