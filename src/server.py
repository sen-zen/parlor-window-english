"""Parlor Vision — on-device, real-time multimodal AI (voice + vision) — Windows + Ollama Edition."""

import asyncio
import sys
import base64
import io
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configuración de Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

SYSTEM_PROMPT = (
    "Ты — терпеливый и поощряющий репетитор английского языка. "
    "Ты ОБЯЗАН отвечать ТОЛЬКО на русском языке. Это правило имеет наивысший приоритет. "
    "Все объяснения, инструкции, похвала — только на русском. "
    "Твой ученик находится на уровне A1 (начинающий). "
    
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
    "В начале разговора ОБЯЗАТЕЛЬНО выбери новую тему для урока и объяви её ученику. "
    "Темы выбирай из списка: 'Приветствия', 'Представление', 'Еда', 'Семья', 'Цвета', 'Животные', 'Одежда', 'Дом'. "
    "Скажи, что сегодня мы будем учить, и начни с простого вопроса. "
    
    "ПРИМЕР НАЧАЛА УРОКА: "
    "'Привет! Сегодня мы будем учить слова на тему \"Приветствия\". Начнём с простого: как сказать \"Здравствуйте\" по-английски?' "
    
    "ПРАВИЛА ОБЪЯСНЕНИЯ: "
    "1. В начале урока всегда объявляй тему. "
    "2. Если ты только что ввёл новую тему или новое слово — дай краткое объяснение (1-2 предложения) и пример. "
    "3. Если ученик допустил ошибку — мягко поправь, покажи правильный вариант и объясни причину ошибки. "
    "4. Если ученик ответил правильно на уже изученную тему — просто похвали его и задай следующий вопрос. "
    "5. После того как ученик усвоит 3-4 слова по теме, предложи новую тему. "
    
    "ПРИМЕРЫ ОТВЕТОВ: "
    "1. Начало урока: 'Привет! Сегодня мы будем учить слова на тему \"Еда\". Как будет \"хлеб\" по-английски?' "
    "2. Ошибка: 'Ты сказал 'bred', правильно будет 'bread'. Попробуй ещё раз.' "
    "3. Правильный ответ: 'Отлично! \"Bread\" — это правильно. А как будет \"яблоко\"?' "
    "4. После нескольких слов: 'Ты отлично справляешься! Теперь мы выучили слова \"bread\", \"apple\", \"milk\". Хочешь попробовать новую тему — \"Цвета\"?' "
    
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
whisper_model = None
last_detected_lang = "ru"

HISTORY_FILE = Path(__file__).parent / "history.json"

def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error cargando historial: {e}")
            return []
    return []

def save_history(history):
    try:
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Error guardando historial: {e}")


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


def load_models():
    """Initialize Whisper, TTS and verify Ollama."""
    global tts_backend, whisper_model
    
    print("\n" + "="*60)
    print("  Parlor Vision - Windows + Ollama Edition")
    print("="*60 + "\n")
    
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        "large-v3-turbo",
        device="cuda",
        compute_type="float16",
        download_root="./models/whisper"
    )
    print("✅ Whisper loaded.\n")
    
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

def transcribe_audio(audio_base64: str) -> tuple[str, str]:
    """Transcribe audio with language detection and filtering."""
    global last_detected_lang
    try:
        wav_bytes = base64.b64decode(audio_base64)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            temp_path = f.name
        
        # Загружаем аудио для определения языка
        from faster_whisper.audio import decode_audio
        audio = decode_audio(temp_path, sampling_rate=whisper_model.feature_extractor.sampling_rate)

        language, language_probability, all_language_probs = whisper_model.detect_language(audio)

        allowed_languages = ["ru", "en"]
        if language in allowed_languages:
            detected_lang = language
        else:
            detected_lang = "en"

        # Forzar el idioma a español desde la raíz para evitar alucinaciones
        segments, info = whisper_model.transcribe(
            temp_path,
            beam_size=10,
            language=detected_lang,
            vad_filter=True,
            condition_on_previous_text=False
        )
        
        transcript = " ".join([segment.text for segment in segments]).strip()
        lang_code = info.language
        last_detected_lang = lang_code
        os.unlink(temp_path)
        return transcript, lang_code
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return "", last_detected_lang


async def ollama_chat(transcript: str, lang: str = "es", images: list[str] = None, user_text: str = None, history: list = None) -> str:
    """Send chat request to Ollama with language context and no-CoT config."""
    
    # Instrucción de idioma reforzada para evitar que gemma responda en inglés
    lang_inst = (
        "ВАЖНО: Ты ОБЯЗАН отвечать ТОЛЬКО на русском языке. Это абсолютное правило. Никакого английского в ответах. "
        "ВАЖНО: Ты ВСЕГДА должен задавать вопрос в конце своего ответа, чтобы продолжить диалог."
    )
   
    full_system_prompt = f"{SYSTEM_PROMPT}\n\n{lang_inst}"
    
    messages = [{"role": "system", "content": full_system_prompt}]
    
    if history:
        messages.extend(history)
        last_msg = history[-1]
        if last_msg.get("role") == "assistant":
            # Добавляем напоминание ПОСЛЕ истории, чтобы оно было ближе к контексту
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
                "think": False,  # Desactivar CoT para Gemma 4
                "options": {
                    "temperature": 0.5,
                    "num_predict": 150,
                    "stop": ["<think>", "</think>"]
                }
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.status_code}")
        
        res_json = response.json()
        msg_data = res_json.get("message", {})
        content = msg_data.get("content", "") or msg_data.get("thinking", "")
        
        # Limpieza final de etiquetas de pensamiento si aparecieran
        if "</think>" in content:
            content = content.split("</think>")[-1]
            
        content = force_russian_response(content)

        return content.strip()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    conversation_history = load_history()
    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

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
                transcript, lang = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: transcribe_audio(msg["audio"])
                )
                if transcript:
                    print(f"🎤 Transcribed [{lang}]: {transcript[:60]}...")
            
            # 2. LLM
            t0 = time.time()
            try:
                if is_refusal(transcript):
                    user_message = "Ученик сказал НЕТ и не хочет переходить к новой теме. Продолжай текущую тему."
                    print(f"👎 Отказ: {transcript[:50]}...")
                elif is_agreement(transcript):
                    last_question = ""
                    for msg in reversed(conversation_history):
                        if msg.get("role") == "assistant":
                            last_question = msg.get("content", "")
                            break
                    
                    tema_words = r'\b(тема|теме|тему)\b'
                    if re.search(tema_words, last_question):
                        user_message = "Ученик СОГЛАСЕН и хочет перейти к новой теме."
                    else:
                        user_message = "Ученик СОГЛАСЕН с твоим предложением. Продолжай."
                    print(f"👍 Согласие: {transcript[:50]}...")
                else:
                    # 🔥 ПРОВЕРКА НА "НЕ-ОТВЕТЫ"
                    last_assistant_msg = ""
                    for msg in reversed(conversation_history):
                        if msg.get("role") == "assistant":
                            last_assistant_msg = msg.get("content", "")
                            break
                    
                    # Проверяем, был ли задан вопрос
                    is_question = False
                    question_markers = r'\b(по-английски|как сказать|попробуй|скажи|назови)\b'
                    if re.search(question_markers, last_assistant_msg):
                        is_question = True
                    
                    if is_question:
                        user_message = f"Ученик сказал: \"{transcript}\". Это его ответ на твой предыдущий вопрос. Проверь, является ли это правильным ответом."
                        print(f"⚠️ Ответ на вопрос: {transcript[:50]}...")
                    else:
                        user_message = transcript
                        print(f"💬 Сообщение: {transcript[:50]}...")

                conversation_history.append({"role": "user", "content": user_message or transcript or "..."})

                text_response = await ollama_chat(
                    transcript=transcript,
                    lang=lang,
                    images=[msg["image"]] if msg.get("image") else None,
                    user_text=msg.get("text"),
                    history=conversation_history
                )
                llm_time = time.time() - t0
                print(f"✅ LLM ({llm_time:.2f}s): {text_response[:100]}...")
                
                # Actualizar historial
                conversation_history.append({"role": "assistant", "content": text_response})

                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                
                # Autoguardado silencioso de la memoria en disco
                save_history(conversation_history)
                    
            except Exception as e:
                print(f"❌ LLM error: {e}")
                text_response = "Error de procesamiento." if lang == "ru"else "Processing error."
                llm_time = 0

            if interrupted.is_set(): continue

            # ENVIAR TEXTO
            await ws.send_text(json.dumps({
                "type": "text", 
                "text": text_response, 
                "llm_time": round(llm_time, 2),
                "transcription": transcript,
                "language": lang
            }))

            # 3. TTS
            sentences = split_sentences(text_response)
            if not sentences: continue

            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            for i, sentence in enumerate(sentences):
                if interrupted.is_set(): break
                try:
                    pcm = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sentence: tts_backend.generate(s, lang=lang)
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
