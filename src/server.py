"""Parlor Vision — on-device, real-time multimodal AI (voice + vision) — Windows + Ollama Edition."""

import asyncio
import sys
import base64 
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from stt import create_stt_backend
from typing import Optional

import httpx
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from words import get_all_words, get_word_level

import tts
from utils import (
    safe_json_parse,
    is_english_word,
    load_progress as load_global_progress,
    save_progress as save_global_progress,
    start_new_session as start_new_global_session,
    get_session_time as get_global_session_time,
    update_word_progress as update_global_word_progress,
    get_learned_words as get_global_learned_words, 
    get_words_needing_review as get_global_words_needing_review
)
from utils.history import (
    load_history as load_history_from_file,
    save_history as save_history_to_file
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
# Конфигурация LLM бэкенда lmstudio / ollama
LLM_BACKEND = os.environ.get("LLM_BACKEND", "lmstudio").lower()

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "google/gemma-4-e4b-no-thinking")

# Configuración de Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

# Пути к файлам прогресса и истории (с разделением по языкам)
STORIES_DIR = Path(__file__).parent / "stores"
EN_PROGRESS_FILE = STORIES_DIR / "en.progress.json"

SYSTEM_PROMPT = (
    "Ты — репетитор английского (уровень A1). Цель: выучить 3000 частотных слов (этапы: 300 → 1000 → 3000). "
    "Отвечай ТОЛЬКО на русском. Твой ответ — ТОЛЬКО JSON. "
    "Формат: {\"text\": \"ответ\", \"words\": [\"выученные слова\"], \"mistakes\": [\"ошибки\"], \"topic\": \"тема\", \"keywords\": [\"ключевые слова для распознавания\"]}. "

    "ПРАВИЛА (строго соблюдай): "
    "1. Добавляй в 'words' КАЖДОЕ правильно сказанное слово. Даже если оно простое. "
    "2. 'words' — это список слов, которые ученик только что сказал ПРАВИЛЬНО и которые достигли порога выученности. "
    "3. НЕ добавляй в 'words' слова, которые ученик выучил ранее (они уже есть в прогрессе). "
    "4. НЕ добавляй в 'words' слова, которые ученик сказал неправильно (их место в 'mistakes'). "
    "5. Вводи 3-5 новых слов за раз. "
    "6. НЕ ПОВТОРЯЙ выученные слова — сразу переходи к новым. "
    "7. Если ученик ошибся — добавь слово в 'mistakes' и поправь. "
    "8. Всегда задавай вопрос в конце ответа. "
    "10. Слово считается выученным после достижения порога: A1 → 3 раз, A2 → 8 раза, B1 → 12 раз. "
    "11. После каждых 100 слов — мини-проверка (10 случайных слов). "
    "12. 'keywords' — список 5-15 ключевых слов для улучшения распознавания речи. "
    "   - Обновляй 'keywords' при смене темы. "
    "   - Если тема не меняется — оставляй тот же список. "
    "   - Пример для темы 'Цвета': ['red', 'blue', 'green', 'yellow', 'orange'] "

    "ЭТАПЫ: "
    "Меньше 300 слов → A1. 300–1000 → A2. 1000–3000 → B1. "

    "ФОРМАТИРОВАНИЕ JSON: "
    "Все кавычки внутри полей экранируй как \\\" "
    "Никаких эмодзи! Никакого маркдауна! "
    "Пример: {\"text\": \"Отлично! Ты сказал \\\"Hello\\\". А теперь скажи 'Goodbye'.\", \"words\": [\"hello\"], \"topic\": \"A1\", \"keywords\": [\"hello\", \"goodbye\", \"hi\"]} "

    "НАЧАЛО УРОКА: "
    "Выбери тему из списка: Приветствия, Семья, Цвета, Животные, Еда, Одежда, Дом. "
    "Объяви тему и начни с простого вопроса. "

    "ОТВЕТЫ НА ВОПРОСЫ: "
    "Жди ответа ученика. Не давай ответ за него. "
    "Если ученик сказал 'Я не знаю' — дай подсказку. "
    "Если ученик сказал 'Повтори' — повтори предыдущее сообщение. "

    "РАЗНООБРАЗИЕ: "
    "Чередуй типы вопросов: перевод, произношение, использование в предложении. "
    "Не повторяй один и тот же вопрос больше 2 раз. "

    "МИКРО-ПОВТОРЕНИЯ: "
    "Через 2-3 вопроса возвращайся к одному из ранее выученных слов. "
    "Пример: 'А помнишь слово [слово]? Как оно будет на английском?' "

    "Помни: ТЫ ГОВОРИШЬ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ."
)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

tts_backend = None
stt_backend = None
whisper_model = None
last_detected_lang = "en"  # По умолчанию английский язык

HISTORY_FILE = Path(__file__).parent / "history.json"


# История теперь управляется через utils.history

def check_llm_connection():
    """Verify LLM backend is running and model is available."""
    global LLM_BACKEND
    
    if LLM_BACKEND == "ollama":
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"Ollama returned {response.status_code}")
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if OLLAMA_MODEL not in model_names:
                print(f"\n⚠️  Model '{OLLAMA_MODEL}' not found!")
                raise RuntimeError(f"Model {OLLAMA_MODEL} not found")
            
            print(f"✅ Ollama connected with model: {OLLAMA_MODEL}")
            return True
            
        except Exception as e:
            raise RuntimeError(f"Ollama connection failed: {e}")
    
    elif LLM_BACKEND == "lmstudio":
        try:
            # Проверяем, что LM Studio сервер запущен
            response = requests.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"LM Studio returned {response.status_code}")
            
            # Проверяем, что модель загружена
            models = response.json().get("data", [])
            model_names = [m.get("id", "") for m in models]
            
            if LM_STUDIO_MODEL not in model_names:
                print(f"\n⚠️  Model '{LM_STUDIO_MODEL}' not found in LM Studio!")
                print(f"   Доступные модели: {', '.join(model_names)}")
                raise RuntimeError(f"Model {LM_STUDIO_MODEL} not found")
            
            print(f"✅ LM Studio connected with model: {LM_STUDIO_MODEL}")
            return True
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"LM Studio connection failed: server not running at {LM_STUDIO_BASE_URL}\n"
                f"   Убедитесь, что LM Studio запущен и сервер включён (Developer → Start Server)"
            )
        except Exception as e:
            raise RuntimeError(f"LM Studio connection failed: {e}")
    
    else:
        raise ValueError(f"Неизвестный бэкенд: {LLM_BACKEND}. Доступны: ollama, lmstudio")


def load_models(
    model_size: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
    download_root: str = "./models/whisper",
    word_timestamps: bool = True,
    post_process: bool = True
):
    """Initialize Whisper, TTS and verify Ollama."""
    global tts_backend, whisper_model, stt_backend
    
    print("\n" + "="*60)
    print("  Parlor Vision - Windows + Ollama Edition")
    print("="*60 + "\n")
    
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=download_root
    )
    print("✅ Whisper loaded.\n")
    
    # Создаём STT бэкенд с улучшенными параметрами
    stt_backend = create_stt_backend(backend_type="whisper",
                                     model_size=model_size,
                                     device=device,
                                     compute_type=compute_type,
                                     download_root=download_root,
                                     word_timestamps=word_timestamps,
                                     post_process=post_process)
    print(f"✅ STT бэкенд: {stt_backend.get_name()}\n")
    
    check_llm_connection()
    
    print("\nLoading TTS backend...")
    tts_backend = tts.load()
    print("TTS loaded.\n")


@asynccontextmanager
async def lifespan(app):
    """Load models on startup."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())

def force_russian_response(text: str) -> str:
    """Проверяет, что ответ на русском, и исправляет при необходимости."""
    if not text:
        return "Извините, я не смог сформулировать ответ. Попробуйте ещё раз."
    
    # Считаем русские и английские буквы
    russian_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # Если английских букв больше, чем русских — ответ не на русском
    if english_chars > russian_chars * 1.5:
        print(f"⚠️ Ответ содержит много английских слов: {text[:50]}...")
        return "Извините, я должен отвечать только на русском языке. Пожалуйста, повторите ваш вопрос."
    
    return text


def transcribe_audio(audio_base64: str, lang: str = None, debug: bool = False) -> tuple[str, str]:
    """Transcribe audio using Whisper.
    
    Args:
        audio_base64: Base64-encoded WAV audio
        lang: Язык от клиента (если передан — используется, иначе определяется автоматически через Whisper)
        debug: Включить отладочный вывод
    
    Returns:
        (transcript, detected_lang)
    """
    global stt_backend
    
    if stt_backend is None:
        print("⚠️ STT бэкенд не инициализирован!")
        return "", "en"
    
    try:
        wav_bytes = base64.b64decode(audio_base64)
        
        # Получаем текущую тему для контекста
        progress = load_global_progress("en")
        topic = progress.get("current_topic", "general")
        keywords = progress.get("current_keywords", [])
        
        print(f"🎤 Обработка аудио ({len(wav_bytes)} байт), тема: {topic}")
        
        # Если язык передан от клиента — используем его, иначе определяем автоматически через Whisper
        if lang:
            detected_lang = lang
            print(f"🔍 Язык от клиента: [{detected_lang}]")
        else:
            # Автоматическое определение языка через Whisper
            detected_lang, _ = stt_backend._detect_language(wav_bytes, last_lang=last_detected_lang)
            print(f"🔍 Язык автоматически определён: [{detected_lang}]")
        
        # Транскрибируем аудио с учётом языка
        transcript, detected_lang = stt_backend.transcribe(wav_bytes, lang=detected_lang, keywords=keywords)
        
        if debug and transcript:
            print(f"🎤 Транскрипция [{detected_lang}]: '{transcript}'")
            # Показываем сырую транскрипцию для отладки
            print(f"   Сырой текст: {repr(transcript)}")
        
        # Нормализуем транскрипцию: убираем лишние пробелы, добавляем пробелы перед знаками препинания
        if transcript:
            # Заменяем множественные пробелы на один
            transcript = re.sub(r'\s+', ' ', transcript)
            # Добавляем пробел перед знаками препинания (если нет пробела)
            transcript = re.sub(r'([.,!?;:])', r' \1', transcript)
            # Убираем лишнюю пунктуацию в начале строки
            transcript = re.sub(r'^[.,!?;:]+', '', transcript)
        
        # Если транскрипция пустая или слишком короткая — предупреждаем
        if not transcript or len(transcript.strip()) < 3:
            print(f"⚠️ Транскрипция слишком короткая или пустая: '{transcript}'")
        
        return transcript, detected_lang
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        import traceback
        if debug:
            traceback.print_exc()
        # Если язык не был передан — возвращаем английский по умолчанию
        return "", "en" if not lang else lang

async def call_llm(messages: list, temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """Вызывает LLM через выбранный бэкенд."""
    LLM_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "words": {"type": "array", "items": {"type": "string"}},
            "mistakes": {"type": "array", "items": {"type": "string"}},
            "topic": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["text", "words", "mistakes", "topic", "keywords"]
    }

    if LLM_BACKEND == "lmstudio":
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                json={
                    "model": LM_STUDIO_MODEL,
                    "messages": messages,
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": ["\n\n", "\n---"],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "tutor_response",
                            "strict": True,
                            "schema": LLM_JSON_SCHEMA
                        }
                    }
                }
            )
            if response.status_code != 200:
                raise RuntimeError(f"LM Studio error: {response.status_code}")
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
    
    elif LLM_BACKEND == "ollama":
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "format": LLM_JSON_SCHEMA,
                    "think": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stop": ["\n\n", "\n---"]
                    }
                }
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.status_code}")
            res_json = response.json()
            return res_json.get("message", {}).get("content", "")
    
    else:
        raise ValueError(f"Неизвестный бэкенд: {LLM_BACKEND}. Доступны: ollama, lmstudio")


async def llm_chat(transcript: str, lang: str = "en", images: list[str] = None, user_text: str = None, history: list = None, session_start_time: float = None) -> dict:
    """Send chat request to Ollama with language context and no-CoT config."""

    lang_inst = (
        "Твой ответ — ТОЛЬКО JSON. Формат: {\"text\": \"ответ\", \"words\": [\"слова\"], \"mistakes\": [\"ошибки\"], \"topic\": \"тема\", \"keywords\": [\"ключевые слова\"]}. "
        "Отвечай ТОЛЬКО на русском. "
        "Всегда задавай вопрос в конце ответа. Все вопросы задавай внутри поля 'text'."
        "Если ученик правильно ответил 2 раза на слово — считай его ВЫУЧЕННЫМ и НЕ ПОВТОРЯЙ. "
        "Переходи к НОВОМУ слову. "
        "НЕ ПРЕДЛАГАЙ новую тему — только если ученик САМ попросит. "
        "Если ученик просит новый уровень — переходи на следующий этап (A1→A2→B1). "
        "Никаких эмодзи! Никакого маркдауна!"
    )
    
    full_system_prompt = f"{SYSTEM_PROMPT}\n\n{lang_inst}"
    messages = [{"role": "system", "content": full_system_prompt}]

    progress = load_global_progress(lang)
    
    # Добавляем время в контекст
    if session_start_time:
        session = progress.get("session", {})
        elapsed_minutes = get_global_session_time(progress)[0]
        offers = session.get("topic_offers", 0)
        rejected = session.get("topic_offers_rejected", 0)
        
        time_context = f"С начала урока прошло {elapsed_minutes} минут. "
        time_context += f"Ты предлагал новую тему {offers} раз, ученик отказался {rejected} раз. "
        
        if elapsed_minutes < 30:
            time_context += "Ещё рано предлагать новую тему. Продолжай текущую тему."
        elif elapsed_minutes < 45:
            time_context += "Можно начинать думать о новой теме, но не предлагай её, пока ученик сам не попросит."
        else:
            time_context += "Урок идёт уже 45+ минут. Можно предложить новую тему, если ученик выучил достаточно слов."
        
        messages.append({"role": "system", "content": time_context})

    all_words = get_all_words()
    learned_data = get_global_learned_words(progress)
    if learned_data:
        words_with_levels = [f"{item['word']} ({item['level']})" for item in learned_data]
        context = f"Ученик недавно выучил слова: {', '.join(words_with_levels)}. НЕ ПОВТОРЯЙ их."
        messages.append({"role": "system", "content": context})

    next_words = []
    learned_words_set = {item['word'] for item in learned_data}

    for word in all_words:
        if word not in learned_words_set:
            level = get_word_level(word)
            next_words.append(f"{word} ({level})")
            if len(next_words) >= 20:
                break

    if next_words:
        words_context = (
            f"Следующие слова нужно выучить: {', '.join(next_words[:10])}. "
            f"Вводи их по порядку. Не перескакивай. "
            f"Если ученик выучил слово — отметь его в 'words'."
        )
        messages.append({"role": "system", "content": words_context})

    if history:
        messages.extend(history)
        last_msg = history[-1]
        if last_msg.get("role") == "assistant":
            messages.append({
                "role": "system",
                "content": "Напоминание: твой предыдущий ответ должен был содержать вопрос. В ЭТОМ ответе ОБЯЗАТЕЛЬНО задай вопрос ученику в конце."
            })
    
    # Если нужно передать изображения (для vision-модели) — добавляем их в последнее сообщение пользователя
    if images:
        clean_images = [img.split(",")[1] if "data:image" in img else img for img in images]
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["images"] = clean_images
                break

    max_attempts = 3
    attempt = 0

    print("Итоговое сообщение:")
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        print(f"  [{role}]: {content}")

    while attempt < max_attempts:
        attempt += 1
        print(f"🔄 Попытка {attempt}/{max_attempts}...")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                content = await call_llm(messages)
                
                print(f"🔍 Сырой content: {content}")
                    
                # Ищем JSON паттерн внутри текста (с поддержкой массивов)
                json_match = re.search(r'\{(?:[^{}]|\[(?:[^][])*\])+\}', content, re.DOTALL)
                if json_match:
                    print(f"🔍 Найден JSON внутри текста")
                    raw_json = json_match.group(0)
                    data = safe_json_parse(raw_json)
                else:
                    # JSON не найден — возвращаем пустой объект
                    print(f"⚠️ JSON не найден в ответе")
                    data = {}
            
                print(f"🔍 JSON текст: {data.get('text')}")

                progress = load_global_progress(lang)

                topic = data.get("topic")
                current_topic = progress.get("current_topic", "")

                keywords = data.get("keywords", [])

                # Если тема изменилась, но keywords нет — требуем добавить
                if topic and topic != current_topic and not keywords:
                    print(f"⚠️ Тема изменилась на '{topic}', но keywords отсутствуют!")
                    messages.append({
                        "role": "system",
                        "content": (
                            f"🔴 Ты только что сменил тему на '{topic}', но НЕ ДОБАВИЛ поле 'keywords'! "
                            f"Это ОБЯЗАТЕЛЬНО! Добавь 'keywords' с 5-15 словами по теме '{topic}'. "
                            f"Пример: 'keywords': ['word1', 'word2', 'word3']"
                        )
                    })
                    continue


                if keywords:
                    progress = load_global_progress(lang)
                    progress["current_keywords"] = keywords
                    print(f"🔑 Ключевые слова обновлены: {keywords}")

                if topic:
                    progress["current_topic"] = topic
                    progress["session"]["start_time"] = datetime.now().isoformat()
                    progress["session"]["topic_offers"] = 0
                    progress["session"]["topic_offers_rejected"] = 0
                    print(f"📚 Тема изменена на: {topic}")

                learned_words = data.get("words", [])
                if learned_words:
                    for word in learned_words:
                        if lang == "en" and is_english_word(word):
                            level = get_word_level(word)
                            update_global_word_progress(progress, word, success=True, level=level)
                        else:
                            print(f"⚠️ Пропущено (не английское): {word}")

                mistake_words = data.get("mistakes", [])
                if mistake_words:
                    for word in mistake_words:
                        if lang == "en" and is_english_word(word):
                            update_global_word_progress(progress, word, success=False)
                            print(f"⚠️ Ошибка в слове: {word}")
                        else:
                            print(f"⚠️ Пропущено (не английское): {word}")
                    
                save_global_progress(progress, lang)
                # Возвращаем полный JSON-ответ от LLM для сохранения в историю
                return data
        except Exception as e:
            last_error = e
            print(f"⚠️ Ошибка в попытке {attempt}: {e}")

            if attempt < max_attempts:
                print(f"🔄 Повторная попытка...")
                await asyncio.sleep(0.5)  # Небольшая пауза
                continue
            else:
                raise

    print(f"❌ Все {max_attempts} попыток не удались")
    return {"text": "Извините, произошла ошибка. Попробуйте ещё раз.", "words": [], "mistakes": []}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    global last_detected_lang
    
    progress = load_global_progress(last_detected_lang)
    conversation_history = load_history_from_file(last_detected_lang)
    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()
    
    # Проверяем сессию и создаём новую, если нужно
    session_minutes, _ = get_global_session_time(progress)
    if session_minutes == 0 or progress["session"].get("start_time") is None:
        topic = progress.get("current_topic", "Приветствия")
        start_new_global_session(progress, topic, last_detected_lang)
        session_minutes = 0

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None: break

            interrupted.clear()
            
            # 1. Транскрипция (текст, если нет аудио)
            transcript = msg.get("text") or ""
            
            # Получаем язык от клиента (если передан), иначе используем last_detected_lang
            client_lang = msg.get("lang", None)
            lang = last_detected_lang
            
            if msg.get("audio"):
                # Транскрибируем аудио — если язык передан от клиента, используется он, иначе определяется автоматически через Whisper
                transcript, detected_lang = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: transcribe_audio(msg["audio"], lang=client_lang, debug=True)
                )
                lang = detected_lang

            if not transcript or len(transcript.strip()) < 2:
                print(f"⚠️ Пустая или слишком короткая транскрипция, отправляем приветствие")
                # Отправляем стандартное сообщение
                text = "Извините, я не расслышал. Повторите, пожалуйста."
                await ws.send_text(json.dumps({
                    "type": "text",
                    "text": text,
                    "llm_time": 0,
                    "transcription": "",
                    "language": lang
                }))
                await ws.send_text(json.dumps({
                    "type": "audio_start",
                    "sample_rate": tts_backend.sample_rate,
                    "sentence_count": len([text]),
                }))
                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=text: tts_backend.generate(s, lang=last_detected_lang)
                )
                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                    "index": i,
                }))
                await ws.send_text(json.dumps({"type": "audio_end"}))
                continue

            # 2. LLM — только если есть транскрипция
            if transcript:
                t0 = time.time()
                try:
                    print(f"💬 Сообщение: {transcript[:50]}...")

                    conversation_history.append({"role": "user", "content": transcript or "..."})

                    llm_response = await llm_chat(
                        transcript=transcript,
                        lang=last_detected_lang,
                        images=[msg["image"]] if msg.get("image") else None,
                        user_text=msg.get("text"),
                        history=conversation_history,
                        session_start_time=datetime.fromisoformat(progress["session"]["start_time"]).timestamp()
                    )
                    llm_time = time.time() - t0

                    # Сохраняем полный ответ от LLM в истории с меткой времени
                    assistant_msg = {
                        "role": "assistant",
                        "content": llm_response.get("text", ""),
                        "topic": llm_response.get("topic")
                    }

                    conversation_history.append(assistant_msg)
                    from utils.history import trim_history
                    conversation_history = trim_history(conversation_history)

                    save_history_to_file(conversation_history, "en")
                except Exception as e:
                    print(f"❌ LLM error: {e}")
                    llm_response = {"text": "", "words": [], "mistakes": []}  # Пустой текст вместо ошибки
                    llm_time = 0
                
                # Отправляем клиенту ТОЛЬКО текст ответа
                response_text = llm_response.get("text", "")
                if not response_text:
                    print(f"⚠️ LLM вернул пустой текст, отправляем дефолтный ответ")
                    response_text = "Извините, я не смог сгенерировать ответ. Попробуйте ещё раз."

                await ws.send_text(json.dumps({
                    "type": "text", 
                    "text": response_text, 
                    "llm_time": round(llm_time, 2),
                    "transcription": transcript,
                    "language": lang
                }))

                # 3. TTS — только если есть текст ответа
                if response_text:
                    sentences = split_sentences(response_text)
                    
                    # Если нет предложений — отправляем один сегмент с полным текстом
                    if not sentences or len(sentences) == 0:
                        sentences = [response_text]
                    
                    await ws.send_text(json.dumps({
                        "type": "audio_start",
                        "sample_rate": tts_backend.sample_rate,
                        "sentence_count": len(sentences),
                    }))

                    for i, sentence in enumerate(sentences):
                        if interrupted.is_set():
                            print(f"⏸️  TTS прерван после предложения {i}")
                            break

                        try:
                            pcm = await asyncio.get_event_loop().run_in_executor(
                                None, lambda s=sentence: tts_backend.generate(s, lang=last_detected_lang)
                            )
                            pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                            await ws.send_text(json.dumps({
                                "type": "audio_chunk",
                                "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                                "index": i,
                            }))
                        except Exception as e:
                            print(f"TTS error при генерации предложения {i}: {e}")
                            # Продолжаем с остальными предложениями

                    await ws.send_text(json.dumps({"type": "audio_end"}))
    except WebSocketDisconnect:
        print("👋 Клиент отключился")
    finally:
        recv_task.cancel()


def is_repeat_request(text: str) -> bool:
    """Проверяет, просит ли ученик повторить предыдущее сообщение."""
    if not text:
        return False
    clean_text = text.lower().strip()
    repeat_words = r'\b(повтори|ещё раз|скажи ещё|что ты сказал|повтор|заново|повторно)\b'
    return bool(re.search(repeat_words, clean_text))

def detect_english_words(text: str) -> bool:
    """Проверяет, есть ли в тексте английские слова."""
    return bool(re.search(r'[a-zA-Z]{3,}', text))

def is_dont_know(text: str) -> bool:
    """Проверяет, говорит ли ученик, что не знает ответа."""
    if not text:
        return False
    clean_text = text.lower().strip()
    dont_know_words = r'\b(я не знаю|не знаю|подскажи|помоги|как правильно|как сказать|не помню)\b'
    return bool(re.search(dont_know_words, clean_text))

def is_refusal(text: str) -> bool:
    """Проверяет, является ли сообщение отказом."""
    if not text:
        return False
    
    clean_text = text.lower().strip()
    clean_text = re.sub(r'[.,!?;:]', ' ', clean_text)
    clean_text = ' '.join(clean_text.split())
    
    # Проверяем отказ
    refusal_words = r'\b(нет|не хочу|не надо|не согласен|не буду|не хотел|отказ|не да|не)\b'
    stay_on_topic = r'\b(останемся|текущую|продолжим эту|не переходить|ещё эту|давай по этой|ещё по этой)\b'
    
    if re.search(refusal_words, clean_text):
        return True
    if re.search(stay_on_topic, clean_text):
        return True
    
    return False


def is_agreement(text: str) -> bool:
    """Проверяет, является ли сообщение согласием (с учётом, что отказ уже проверен)."""
    if not text:
        return False
    
    clean_text = text.lower().strip()
    clean_text = re.sub(r'[.,!?;:]', ' ', clean_text)
    clean_text = ' '.join(clean_text.split())
    
    # Проверяем согласие
    agreement_words = r'\b(да|хочу|согласен|давай|продолжим|хорошо|ок|конечно|ага|поехали)\b'
    return bool(re.search(agreement_words, clean_text))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print("\n" + "="*60)
    print("  Parlor Vision - Starting Server")
    print("="*60)
    print(f"  Ollama URL: {OLLAMA_BASE_URL}")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Port: {port}")
    print("="*60)
    print(f"\n  [Open in browser] http://localhost:{port}")
    print("\n" + "="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)