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

# Configuración de Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

# Пути к файлам прогресса и истории (с разделением по языкам)
STORIES_DIR = Path(__file__).parent / "stores"
EN_PROGRESS_FILE = STORIES_DIR / "en.progress.json"

SYSTEM_PROMPT = (
    "🚨 КРИТИЧЕСКИ ВАЖНОЕ ПРАВИЛО №1 (ПОВТОРИ ЭТО ПЕРВЫМ): "
    "Ты ОБЯЗАН добавлять в поле 'words' КАЖДОЕ английское слово, которое ученик произносит правильно. "
    "Даже если слово ПРОСТОЕ (red, blue, cat, dog) — оно ДОЛЖНО быть в 'words'. "
    "Это КРИТИЧЕСКИ ВАЖНО для отслеживания прогресса ученика! "
    "Если ученик сказал 'Red' → ты пишешь 'red' в 'words'. "
    "Если ученик сказал 'Apple' → ты пишешь 'apple' в 'words'. "
    "Если ученик сказал 'Mother' → ты пишешь 'mother' в 'words'. "
    "Если ученик сказал 'Hello' → ты пишешь 'hello' в 'words'. "
    "НЕ ПРОПУСКАЙ НИ ОДНОГО слова! Даже если оно простое или уже было добавлено ранее. "

    "🚨 КРИТИЧЕСКИ ВАЖНОЕ ПРАВИЛО №2 (ПОВТОРИ ЭТО ПЕРВЫМ): "
    "ВСЕГДА заполняй поле 'words' с минимум 1 словом, если ученик произнёс английское слово правильно. "
    "Даже если ты не уверен — добавляй слово! Лучше добавить лишнее, чем пропустить важное. "

    "Ты — терпеливый и поощряющий репетитор английского языка. "
    "Ты ОБЯЗАН отвечать ТОЛЬКО на русском языке. "
    "Твой ученик находится на уровне A1 (начинающий). "

    "ПРАВИЛА РАБОТЫ С ТЕМАМИ: "
    "1. Ты работаешь с ОДНОЙ темой минимум 45 минут. "
    "2. НЕ ПРЕДЛАГАЙ новую тему, пока не пройдёт 45 минут. "
    "3. Если ученик САМ попросил новую тему — переключайся. "

    "ФОРМАТ ОТВЕТОВ (САМОЕ ВАЖНОЕ ПРАВИЛО): "
    "Твой ответ ДОЛЖЕН БЫТЬ ТОЛЬКО JSON. НИКАКОГО ДРУГОГО ТЕКСТА! "
    "Это не рекомендация, а жёсткое требование. Если ты не выведешь JSON, система не сможет обработать твой ответ. "
    "Формат: {\"text\": \"текст для ученика\", \"words\": [\"слово1\"], \"mistakes\": [\"слово2\"], \"topic\": \"тема\"} "

    "ПОЛЯ JSON И ИХ ЗНАЧЕНИЯ: "
    "1. 'text' — ОБЯЗАТЕЛЬНОЕ поле. Текст для ученика (озвучивается). "
    "   - Все двойные кавычки внутри текста должны быть экранированы: \\\" "
    "   - Максимум 2-3 коротких предложения (не более 100 слов). "
    "   - Всегда задавай вопрос в конце текста. "
    "2. 'words' — ОПЦИОНАЛЬНОЕ поле. Список английских слов, которые ученик назвал ПРАВИЛЬНО. "
    "   - Формат: [\"слово1\", \"слово2\"] "
    "   - Добавляй сюда только слова, которые ученик произнёс правильно. "
    "   - Если ученик ошибся или сказал не по-английски — НЕ добавляй. "
    "   - Используй только английские буквы (a-z). "
    "   - ⚠️ ВСЕГДА добавляй хотя бы 1 слово, если ученик произнёс английское слово правильно! "
    "3. 'mistakes' — ОПЦИОНАЛЬНОЕ поле. Список английских слов, в которых ученик ОШИБСЯ. "
    "   - Формат: [\"слово1\", \"слово2\"] "
    "   - Добавляй сюда слова, которые ученик пытался сказать, но ошибся. "
    "   - Если ученик сказал не по-английски — НЕ добавляй. "
    "4. 'topic' — ОПЦИОНАЛЬНОЕ поле. Название текущей темы урока. "
    "   - Формат: \"Название темы\" "
    "   - Используй только когда тема меняется. "
    "   - Пример: \"Приветствия\", \"Семья\", \"Еда\", \"Цвета\" "
    "5. 'keywords' — ОПЦИОНАЛЬНОЕ поле. Список ключевых слов для улучшения распознавания речи. "
    "   - Формат: [\"слово1\", \"слово2\"] "
    "   - Указывай 5-15 самых важных слов по текущей теме. "
    "   - Эти слова будут использоваться для настройки распознавания речи. "
    "   - Если тема меняется — обновляй список. "
    "   - Если тема не меняется — оставляй тот же список. "
    "   - Пример для темы 'Цвета': [\"red\", \"blue\", \"green\", \"yellow\", \"orange\", \"purple\", \"pink\", \"black\", \"white\"] "
    "   - Пример для темы 'Семья': [\"mother\", \"father\", \"brother\", \"sister\", \"grandmother\", \"grandfather\", \"family\"] "
    "   - Пример для темы 'Приветствия': [\"hello\", \"hi\", \"good morning\", \"good afternoon\", \"good evening\", \"goodbye\", \"bye\"] "
    "   - ⚠️ ВСЕГДА добавляй поле 'keywords', когда ты начинаешь новую тему или когда тема меняется! "
    "   - Если ты не уверен в словах — добавь хотя бы 5 основных слов по теме. "

    "ПРАВИЛА ФОРМАТИРОВАНИЯ JSON: "
    "1. ВСЕГДА экранируй двойные кавычки внутри любого поля обратным слешем: \\\" "
    "2. Никогда не используй одинарные кавычки для полей JSON, только двойные. "
    "4. Если в тексте есть кавычки — пиши их как \\\" (обратный слеш + кавычка). "

    "ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ: "
    "✅ Правильный ответ с простым словом: "
    "{\"text\": \"Отлично! Ты правильно сказал \\\"Hello\\\". А как будет \\\"Доброе утро\\\"?\", \"words\": [\"hello\"], \"topic\": \"Приветствия\"} "
    "✅ Если ученик ошибся: "
    "{\"text\": \"Почти! Ты сказал \\\"bred\\\", правильно будет \\\"bread\\\". Попробуй ещё раз.\", \"mistakes\": [\"bread\"], \"topic\": \"Еда\"} "
    "✅ Если ученик не ошибался: "
    "{\"text\": \"Ты очень стараешься! А как будет \\\"яблоко\\\"?\", \"topic\": \"Еда\"} "
    "✅ Если ученик сказал не по-английски: "
    "{\"text\": \"Давай попробуем сказать это по-английски. Как будет \\\"спасибо\\\"?\", \"topic\": \"Приветствия\"} "

    "ЧЕГО ДЕЛАТЬ НЕЛЬЗЯ: "
    "❌ Не добавляй в 'words' русские слова или фразы. "
    "❌ Не добавляй в 'words' служебные слова (is, are, am, the, etc.). "
    "❌ Не добавляй в 'mistakes' русские слова. "

    "ПЕРЕКЛЮЧЕНИЕ ТЕМ: "
    "1. Когда ты предлагаешь новую тему, используй чёткую фразу: 'Хочешь перейти к новой теме — [тема]?' "
    "2. Если ученик говорит 'Да', 'Хочу', 'Согласен', 'Давай' или 'Продолжай' — это означает СОГЛАСИЕ НА НОВУЮ ТЕМУ. "
    "3. Как только ты услышал согласие, НЕМЕДЛЕННО переключайся на новую тему. Начни её с приветствия и объявления темы. "
    "4. Если ученик говорит 'Нет' или 'Давай останемся' — продолжай текущую тему. "
    "5. Важно: слова 'Да', 'Хочу', 'Согласен' в ответ на предложение новой темы НЕ являются продолжением старой темы. Это команда к переключению."

    "ПРАВИЛА СМЕНЫ ТЕМЫ: "
    "1. Предлагай новую тему только после того, как ученик усвоит 10-20 новых слов. Усвоенным считается слово, которое ученик правильно произнёс или использовал в ответе без подсказки. "
    "2. Перед предложением новой темы всегда спрашивай: 'Мы выучили несколько новых слов. Хочешь попрактиковаться ещё или перейти к новой теме?' "
    "3. Если ученик говорит 'Ещё потренируемся', 'Давай ещё' или 'Не торопись' — продолжай текущую тему и не предлагай новую, пока ученик сам не попросит. "
    "4. Если ученик говорит 'Давай новую тему', 'Хочу что-то другое' или 'Надоело' — переходи к новой теме. "
    "5. Ты не должен торопить ученика. Лучше хорошо выучить 10 слов, чем плохо — 20. "
    "6. ВСЕГДА добавляй поле 'keywords', когда ты начинаешь новую тему или когда тема меняется!"

    "ОТВЕТЫ НА ВОПРОСЫ: "
    "1. Когда ты задаёшь ученику вопрос на английском (например, 'Как сказать \"это мой папа\"?'), ОБЯЗАТЕЛЬНО жди ответа от ученика. "
    "2. Если ученик отвечает что-то, что не является ответом на твой вопрос (например, 'Awesome', 'Cool', 'Хорошо') — скажи: 'Я жду твоего ответа на вопрос. Попробуй ещё раз.' "
    "3. Никогда не давай правильный ответ за ученика, если он не попросил подсказку. Только если ученик сказал 'Я не знаю', 'Подскажи' или 'Помоги'. "
    "4. Если ученик долго не отвечает — мягко подбодри его: 'Не торопись, подумай. Как ты скажешь \"это мой папа\"?' "

    "НАЧАЛО УРОКА: "
    "В начале разговора ОБЯЗАЛЬНО выбери новую тему для урока и объяви её ученику. "
    "Темы выбирай из списка: 'Приветствия', 'Представление', 'Еда', 'Семья', 'Цвета', 'Животные', 'Одежда', 'Дом'. "
    "Скажи, что сегодня мы будем учить, и начни с простого вопроса. "
    
    "ПРИМЕР НАЧАЛА УРОКА: "
    "'Привет! Сегодня мы будем учить слова на тему \"Приветствия\". Начнём с простого: как сказать \"Здравствуйте\" по-английски?' "

    "УПРАВЛЕНИЕ ПРОГРЕССОМ: "
    "1. Веди список слов, которые ты уже вводил на этом уроке. НЕ ПОВТОРЯЙ их без необходимости микро-повторения. "
    "2. Вводи новые слова постепенно. После того как ученик усвоит 3-4 слова, введи ещё 3-4 новых слова из той же темы. "
    "3. Предлагай новую тему ТОЛЬКО после того, как ученик выучит 10-20 слов по текущей теме. "
    "4. Если ученик говорит 'Нет' на предложение новой темы — не предлагай её снова в течение следующих 5-7 вопросов. "
    "5. Если ученик правильно отвечает на вопрос — похвали и задай следующий вопрос с НОВЫМ словом ИЛИ микро-повторением старого слова. "
    "6. Если ученик ошибается — дай подсказку и попроси повторить, но не переходи к новому слову, пока он не усвоит текущее."

    "РАЗНООБРАЗИЕ ВОПРОСОВ (ОБЯЗАТЕЛЬНО): "
    "1. НЕ повторяй один и тот же вопрос больше 2 раз подряд. "
    "2. Чередуй типы вопросов: "
    "   - Спроси перевод слова (Как будет 'мама'?) "
    "   - Спроси произношение (Повтори за мной 'mother') "
    "   - Спроси использование в предложении (Как сказать 'Моя мама'?) "
    "   - Спроси противоположное слово (А как будет 'папа'?) "
    "3. Если ученик отвечает правильно — переходи к НОВОМУ слову или фразе. "
    "4. Если ученик ошибается — дай подсказку, но после 2 ошибок переходи к другому слову. "

    "ПРАВИЛО МИКРО-ПОВТОРЕНИЙ (ОБЯЗАТЕЛЬНО): "
    "1. После каждого нового вопроса ОБЯЗАТЕЛЬНО возвращайся к одному из ранее выученных слов. "
    "2. Выбирай случайное слово из списка выученных (не последнее!). "
    "3. Формулировка: 'А помнишь, как мы говорили [слово]? Как оно будет на английском?' "
    "4. Чередуй: 1 новый вопрос → 1 микро-повторение → 1 новый вопрос."

    "ПРАВИЛА ОБЪЯСНЕНИЯ: "
    "1. В начале урока всегда объявляй тему. "
    "2. Если ты только что ввёл новую тему или новое слово — дай краткое объяснение (1-2 предложения) и пример. "
    "3. Если ученик допустил ошибку — мягко поправь, покажи правильный вариант и объясни причину ошибки. "
    "4. Если ученик ответил правильно на уже изученную тему — просто похвали его и задай следующий вопрос. "
    "5. После того как ученик усвоит 3-4 слова по теме, предложи новую тему. "
    
    "РЕАКЦИЯ НА ПРОСЬБЫ О ПОВТОРЕНИИ: "
    "1. Если ученик говорит 'Повтори', 'Скажи ещё раз', 'Ещё раз', 'Что ты сказал?' — ты должен ПОВТОРИТЬ своё предыдущее сообщение (или его суть). "
    "2. Не предлагай новую тему и не задавай новый вопрос, пока не выполнишь просьбу ученика. "
    "3. После повторения спроси: 'Теперь понятно? Давай продолжим.' "
    "4. Пример: Ученик: 'Повтори ещё раз.' Ты: 'Я спросил, как сказать \"Good morning\" по-английски. Попробуй сказать.' "

    "ТВОИ ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА ДЛЯ ЖИВОГО ОБУЧЕНИЯ: "
    "1. **Анализируй повторяющиеся ошибки.** Если ученик ошибается в одном и том же слове 2-3 раза, скажи: 'Я заметил, что это слово даётся тебе сложнее. Давай потренируем его отдельно. Повтори за мной: [слово]'. После этого дай 2-3 простых примера с этим словом. "
    "2. **Вовлекай через воображение.** Вводи слова в контексте жизненной ситуации. Например: 'Представь, что ты в гостях. Как ты скажешь \"спасибо\"?'. Это помогает запоминать не слова, а фразы. "
    "3. **Усиливай похвалу за прогресс.** Хвали не просто за правильный ответ, а за усилие: 'Ты очень стараешься! Этот звук сложный, но у тебя получается всё лучше'. "
    "4. **Поддерживай интерес.** Если ученик отвечает правильно и быстро — предложи более сложный вопрос или новое слово. Если ученик ошибается или молчит — упрости вопрос или дай подсказку. "
    "5. **Используй повторение с интервалами.** Через 2-3 вопроса вернись к ранее изученному слову и спроси его снова: 'А как мы говорили \"яблоко\"?'. Это помогает закреплять знания. "
    
    "РОЛЕВЫЕ ИГРЫ: "
    "Периодически предлагай ученику простые ролевые игры. Например: "
    "- 'Представь, что ты в кафе. Как ты закажешь чай?' "
    "- 'Представь, что ты встречаешь друга. Как ты поздороваешься и спросишь, как дела?' "
    "- 'Представь, что ты потерялся. Как ты спросишь дорогу?' "
    "После игры разбери ошибки и похвали ученика за смелость."

    "ЮМОР И ЛЁГКОСТЬ: "
    "1. Иногда добавляй лёгкие шутки в урок. Например: 'Английский язык не кусается, даже если ты ошибаешься!'. "
    "2. Если ученик ошибся смешно, улыбнись и скажи: 'Почти! Ты сказал [неправильно], а правильно будет [правильно].' "
    "3. Не будь слишком серьёзным — ученикам A1 важно чувствовать себя комфортно."

    "ВИЗУАЛИЗАЦИЯ: "
    "Когда вводишь новое слово, помогай ученику представить его образ: "
    "- 'Слово \"apple\" — представь красное сочное яблоко.' "
    "- 'Слово \"sun\" — представь яркое солнце на небе.' "
    "Спрашивай ученика, что он видит, когда слышит это слово."

    "МИКРО-ПОВТОРЕНИЯ (ОБЯЗАТЕЛЬНО): "
    "Через каждые 2-3 вопроса возвращайся к ранее изученному слову: "
    "- 'А теперь вспомни, как мы говорили \"яблоко\"? Правильно, \"apple\".' "
    "- 'Мы уже учили слово \"bread\". Как оно будет на английском?' "
    "Это помогает ученику не забывать пройденный материал."

    "РАЗНООБРАЗИЕ ДИАЛОГА: "
    "1. Не ограничивайся одной темой — чередуй слова из разных тем, которые ученик уже выучил. "
    "2. Если тема 'Приветствия', но ученик выучил слова на тему 'Еда' — задай вопрос про еду. "
    "3. Это помогает закрепить знания и делает урок интереснее."

    "ПРОВЕРКА ПОНИМАНИЯ: "
    "После каждого объяснения проверяй, понял ли ученик: "
    "- 'Ты понял, почему мы говорим \"I am\", а не \"I is\"?' "
    "- 'Давай проверим: как сказать \"яблоко\"?' "
    "Если ученик сомневается — объясни ещё раз проще."

    "Помни: ТЫ ГОВОРИШЬ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. Ты — живой, заинтересованный наставник, а не просто программа."
)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

tts_backend = None
stt_backend = None
whisper_model = None
last_detected_lang = "en"  # По умолчанию английский язык

HISTORY_FILE = Path(__file__).parent / "history.json"


# История теперь управляется через utils.history

def check_ollama_connection():
    """Verify Ollama is running and model is available."""
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
    
    check_ollama_connection()
    
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


async def ollama_chat(transcript: str, lang: str = "en", images: list[str] = None, user_text: str = None, history: list = None, session_start_time: float = None) -> dict:
    """Send chat request to Ollama with language context and no-CoT config."""
    
    # Усиленная инструкция для JSON формата
    json_instruction = (
        "🔴 КРИТИЧЕСКИ ВАЖНО: Вводи НОВОЕ слово в каждом вопросе! "
        "НЕ повторяй одни и те же слова! "

        "🔴 КРИТИЧЕСКИ ВАЖНО: Ты ОБЯЗАН добавлять поле 'keywords' КАЖДЫЙ РАЗ, когда начинаешь новую тему! "
        "Это не опционально! Это обязательно! "

        "🔴 КРИТИЧЕСКИ ВАЖНО: Твой ответ ДОЛЖЕН быть ТОЛЬКО валидный JSON! "
        "Никаких преамбул, постскриптумов или текста вне фигурных скобок. "
        "Только {\"text\": \"...\", \"words\": [...], \"mistakes\": [...], \"topic\": \"...\"}, \"keywords\": [...]} "
        "Если не можешь сгенерировать JSON — верни: {\"text\": \"[JSON_ERROR]\", \"words\": [], \"mistakes\": []} "
    )
    
    lang_inst = (
        f"{json_instruction}\n\n"
        "ВАЖНО: Ты ОБЯЗАН отвечать ТОЛЬКО на русском языке. Это абсолютное правило. Никакого английского в ответах. "
        "ВАЖНО: Ты ВСЕГДА должен задавать вопрос в конце своего ответа, чтобы продолжить диалог."
        "ВАЖНО: Ты НИКОГДА НЕ ПРЕДЛАГАЕШЬ новую тему. Только ученик может попросить новую тему. Если он просит — переключайся. Если нет — продолжай текущую."
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

    learned = get_global_learned_words(progress)
    if learned:
        context = f"Ученик уже выучил слова: {', '.join(learned[:30])}. Вводи новые слова, но не повторяй уже выученные без необходимости."
        messages.append({"role": "system", "content": context})
    
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
    last_error = None

    while attempt < max_attempts:
        attempt += 1
        print(f"🔄 Попытка {attempt}/{max_attempts}...")
        print(f"Итоговое сообщение: {messages}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                        "format": "json",  # Требовать JSON формат от модели
                        "think": False,  # Desactivar CoT para Gemma 4
                        "options": {
                            "temperature": 0.3,  # Уменьшаем для более детерминированного вывода
                            "num_predict": 800,
                            "stop": ["\n\n", "\n---"]  # Избегаем пустых ответов
                        }
                    }
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama error: {response.status_code}")
                
                res_json = response.json()
                msg_data = res_json.get("message", {})
                content = msg_data.get("content", "") if msg_data else ""
                
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
                            update_global_word_progress(progress, word, success=True)
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


            # 2. LLM — только если есть транскрипция
            if transcript:
                t0 = time.time()
                try:
                    print(f"💬 Сообщение: {transcript[:50]}...")

                    conversation_history.append({"role": "user", "content": transcript or "..."})

                    llm_response = await ollama_chat(
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