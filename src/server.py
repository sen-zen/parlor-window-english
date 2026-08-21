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
    "5. Ты не должен торопить ученика. Лучше хорошо выучить 10 слов, чем плохо — 20."

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
    "1. Веди список слов, которые ты уже вводил на этом уроке. НЕ ПОВТОРЯЙ их без необходимости. "
    "2. Вводи новые слова постепенно. После того как ученик усвоит 3-4 слова, введи ещё 3-4 новых слова из той же темы. "
    "3. Предлагай новую тему ТОЛЬКО после того, как ученик выучит 10-20 слов по текущей теме. "
    "4. Если ученик говорит 'Нет' на предложение новой темы — не предлагай её снова в течение следующих 5-7 вопросов. "
    "5. Если ученик правильно отвечает на вопрос — похвали и задай следующий вопрос с НОВЫМ словом. "
    "6. Если ученик ошибается — дай подсказку и попроси повторить, но не переходи к новому слову, пока он не усвоит текущее."

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

    "МИКРО-ПОВТОРЕНИЯ: "
    "Через каждые 2-3 вопроса возвращайся к ранее изученному слову: "
    "- 'А теперь вспомни, как мы говорили \"яблоко\"? Правильно, \"apple\".' "
    "- 'Мы уже учили слово \"bread\". Как оно будет на английском?' "
    "Это помогает ученику не забывать пройденный материал."

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
    vad_parameters: dict = None,
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
                                     vad_parameters=vad_parameters,
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


def transcribe_audio(audio_base64: str, debug: bool = False) -> tuple[str, str]:
    """Transcribe audio using Whisper with English as default language."""
    global stt_backend
    
    if stt_backend is None:
        print("⚠️ STT бэкенд не инициализирован!")
        return "", "en"
    
    try:
        wav_bytes = base64.b64decode(audio_base64)
        
        # Получаем текущую тему для контекста
        progress = load_global_progress("en")
        topic = progress.get("current_topic", "general")
        
        print(f"🎤 Обработка аудио ({len(wav_bytes)} байт), тема: {topic}")
        
        # По умолчанию английский язык (устанавливается через last_detected_lang)
        detected_lang = last_detected_lang
        
        print(f"🔍 Язык определён: [{detected_lang}]")
        
        # Транскрибируем аудио с учётом языка
        transcript, _ = stt_backend.transcribe(wav_bytes, topic=topic)
        
        if debug and transcript:
            print(f"🎤 Транскрипция [{detected_lang}]: '{transcript}'")
            # Показываем сырую транскрипцию для отладки
            print(f"   Сырой текст: {repr(transcript)}")
        
        # Если транскрипция пустая или слишком короткая — предупреждаем
        if not transcript or len(transcript.strip()) < 3:
            print(f"⚠️ Транскрипция слишком короткая или пустая: '{transcript}'")
        
        return transcript, detected_lang
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        import traceback
        if debug:
            traceback.print_exc()
        return "", "en"  # По умолчанию английский язык


async def ollama_chat(transcript: str, lang: str = "en", images: list[str] = None, user_text: str = None, history: list = None, session_start_time: float = None) -> dict:
    """Send chat request to Ollama with language context and no-CoT config."""
    
    # Усиленная инструкция для JSON формата
    json_instruction = (
        "🔴 КРИТИЧЕСКИ ВАЖНО: Твой ответ ДОЛЖЕН быть ТОЛЬКО валидный JSON! "
        "Никаких преамбул, постскриптумов или текста вне фигурных скобок. "
        "Только {\"text\": \"...\", \"words\": [...], \"mistakes\": [...], \"topic\": \"...\"} "
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

    # Добавляем контекст прогресса ПОСЛЕ создания messages
    learned = get_global_learned_words(progress)
    review_needed = get_global_words_needing_review(progress, days=7)
    if learned:
        context = f"Ученик уже выучил слова: {', '.join(learned[:20])}. "
        if review_needed:
            context += f"Особенно повтори слова: {', '.join(review_needed[:5])}. "
        context += "Вводи новые слова, но не повторяй уже выученные без необходимости."
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
        
        # Пытаемся получить content или thinking
        content = msg_data.get("content", "") if msg_data else ""
        
        try:
            print(f"🔍 Сырой content: {repr(content[:100])}...")
            
            # Ищем JSON паттерн внутри текста
            # Если не нашли — пробуем найти любой JSON
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                print(f"🔍 Найден JSON внутри текста")

                try:
                    raw_json = json_match.group(0)
                    data = safe_json_parse(raw_json)
                except Exception as e:
                    print(f"⚠️ Ошибка парсинга JSON: {e}")
                    print(f"🔍 Сырой ответ: {repr(content[:200])}...")
                    # Fallback — возвращаем текст как есть
                    data = {"text": content, "words": [], "mistakes": []}
            else:
                # Если JSON не найден — пробуем весь текст как JSON
                print(f"⚠️ JSON не найден, пробуем весь текст как JSON")
                data = {"text": content, "words": [], "mistakes": []}

            print(f"🔍 JSON текст: {data.get('text')}")

            progress = load_global_progress(lang)

            topic = data.get("topic")
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
                    update_global_word_progress(progress, word, success=False)
                    print(f"⚠️ Ошибка в слове: {word}")
            
            # Добавляем информацию о выученных словах в историю
            if learned_words:
                learned_str = ", ".join(learned_words)
                # Добавляем системное сообщение о прогрессе
                if history:
                    history.append({
                        "role": "system",
                        "content": f"Ученик выучил слова: {learned_str}"
                    })

            
            save_global_progress(progress, lang)
            # Возвращаем полный JSON-ответ от LLM для сохранения в историю
            return data
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка парсинга JSON: {e}")
            print(f"🔍 Сырой content: {repr(content[:100])}...")
            # Возвращаем текст как fallback
            return {"text": content.strip() if content else "Извините, произошла ошибка.", "words": [], "mistakes": []}
        except Exception as e:
            print(f"⚠️ Неизвестная ошибка при парсинге JSON: {e}")
            print(f"🔍 Сырой content: {repr(content[:100])}...")
            # Возвращаем текст как fallback
            return {"text": content.strip() if content else "Извините, произошла ошибка.", "words": [], "mistakes": []}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    progress = load_global_progress('en')
    conversation_history = load_history_from_file("en")
    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    # Проверяем сессию и создаём новую, если нужно
    session_minutes, _ = get_global_session_time(progress)
    if session_minutes == 0 or progress["session"].get("start_time") is None:
        topic = progress.get("current_topic", "Приветствия")
        start_new_global_session(progress, topic, 'en')
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
            
            # 1. Transcripción
            transcript = ""
            lang = last_detected_lang
            if msg.get("audio"):
                transcript, detected_lang = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: transcribe_audio(msg["audio"], debug=True)
                )
                
            # Определяем язык для сохранения прогресса:
            # если транскрипция содержит английские слова — используем 'en'
            if detected_lang == "ru" and transcript:
                english_chars = len(re.findall(r'[a-zA-Z]{3,}', transcript))
                russian_chars = len(re.findall(r'[а-яА-ЯёЁ]', transcript))
                # Если английских букв больше — это английский язык (даже с русским акцентом)
                if english_chars > russian_chars * 0.5:
                    print(f"🌐 Язык: en (английские слова в транскрипции)")
                    lang = "en"
                    if transcript:
                        print(f"🎤 Transcribed [{lang}]: {transcript[:60]}...")
            
            # 2. LLM
            t0 = time.time()
            try:
                user_message = transcript
                print(f"💬 Сообщение: {transcript[:50]}...")

                conversation_history.append({"role": "user", "content": user_message or transcript or "..."})

                llm_response = await ollama_chat(
                    transcript=transcript,
                    lang=lang,
                    images=[msg["image"]] if msg.get("image") else None,
                    user_text=msg.get("text"),
                    history=conversation_history,
                    session_start_time=datetime.fromisoformat(progress["session"]["start_time"]).timestamp()
                )
                llm_time = time.time() - t0
                print(f"✅ LLM ({llm_time:.2f}s): {llm_response.get('text', '')[:100]}...")

                # Сохраняем полный ответ от LLM в историю с меткой времени
                assistant_msg = {
                    "role": "assistant",
                    "content": llm_response.get("text", ""),
                    "timestamp": datetime.now().isoformat(),
                    "words": llm_response.get("words", []),
                    "mistakes": llm_response.get("mistakes", []),
                    "topic": llm_response.get("topic")
                }
                conversation_history.append(assistant_msg)
                
                save_history_to_file(conversation_history, "en")
            
            except Exception as e:
                print(f"❌ LLM error: {e}")
                llm_response = {"text": "Извините, произошла ошибка. Попробуйте ещё раз.", "words": [], "mistakes": []}
                llm_time = 0
            
            # Отправляем клиенту ТОЛЬКО текст ответа
            await ws.send_text(json.dumps({
                "type": "text", 
                "text": llm_response.get("text", ""), 
                "llm_time": round(llm_time, 2),
                "transcription": transcript,
                "language": lang
            }))

            # 3. TTS
            sentences = split_sentences(llm_response.get("text", ""))
            
            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    break
                try:
                    if detect_english_words(sentence):
                        tts_lang = "en"
                    else:
                        tts_lang = "ru"

                    pcm = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sentence: tts_backend.generate(s, lang=tts_lang)
                    )
                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_text(json.dumps({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                        "index": i,
                    }))
                except Exception as e:
                    print(f"TTS error: {e}")
            
            if not interrupted.is_set():
                await ws.send_text(json.dumps({"type": "audio_end"}))

    except WebSocketDisconnect:
        print("Client disconnected")
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
    print(f"\n  🌐 Open in browser: http://localhost:{port}")
    print("\n" + "="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)