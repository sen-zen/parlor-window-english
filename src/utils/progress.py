from pathlib import Path
from datetime import datetime
import json
import re

# Пути к файлам прогресса по языкам
EN_PROGRESS_FILE = Path(__file__).parent.parent / "stores" / "en.progress.json"
RU_PROGRESS_FILE = Path(__file__).parent.parent / "stores" / "ru.progress.json"


def detect_language_from_transcript(transcript: str) -> str:
    """Определяет язык из транскрипции.
    
    Args:
        transcript: Текст транскрипции
        
    Returns:
        Язык: 'en' или 'ru'
    """
    if not transcript:
        return "ru"  # Fallback на русский
    
    # Считаем английские и русские буквы
    english_chars = len(re.findall(r'[a-zA-Z]{3,}', transcript))
    russian_chars = len(re.findall(r'[а-яА-ЯёЁ]', transcript))
    
    # Если английских букв больше — английский язык
    if english_chars > russian_chars * 1.5:
        return "en"
    
    # Иначе — русский язык
    return "ru"


def default_progress(lang: str = "en") -> dict:
    """Создаёт структуру прогресса для указанного языка."""
    return {
        "learned_words": {},
        "current_topic": "Приветствия",
        "total_sessions": 0,
        "session": {
            "start_time": None,
            "topic_offers": 0,
            "last_topic_offer": None,
            "words_learned_this_session": [],
            "topic_offers_rejected": 0
        },
        "language": lang  # Язык сессии
    }


def load_progress(lang: str = None) -> dict:
    """Загружает прогресс для указанного языка.
    
    Args:
        lang: Язык ("en" или "ru"). Если None — определяется из контекста.
        
    Returns:
        Словарь с прогрессом
    """
    # Определяем язык, если не указан
    if not lang:
        raise ValueError("Язык не указан в прогрессе! Укажите 'language' в load_progress.")
     
    progress_file = EN_PROGRESS_FILE if lang == "en" else RU_PROGRESS_FILE
    print(f"📂 Загрузка прогресса [{lang}] из: {progress_file}")
    print(f"📂 Файл существует: {progress_file.exists()}")
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"📂 Загружено {len(data.get('learned_words', {}))} слов [{lang}]")
                return data
        except Exception as e:
            print(f"⚠️ Ошибка загрузки [{lang}]: {e}")
            return default_progress(lang)
    else:
        print(f"⚠️ Файл [{lang}] не найден, создаю новый")
        return default_progress(lang)


def start_new_session(progress: dict, topic: str, lang: str = None):
    """Начинает новую сессию (обнуляет счётчики).
    
    Args:
        progress: Словарь с прогрессом
        topic: Тема урока
        lang: Язык сессии (опционально)
    """
    if not lang:
        raise ValueError("Язык не указан в прогрессе! Укажите 'language' в load_progress.")
    
    progress["session"] = {
        "start_time": datetime.now().isoformat(),
        "topic_offers": 0,
        "last_topic_offer": None,
        "words_learned_this_session": [],
        "topic_offers_rejected": 0
    }
    progress["current_topic"] = topic
    progress["total_sessions"] += 1
    progress["language"] = lang

    save_progress(progress, lang)
    print(f"📚 Новая сессия [{lang}]: тема '{topic}'")


def reset_session_counters(progress: dict, lang: str = None):
    """Обнуляет счётчики предложений новой темы (без сброса времени)."""
    if not lang:
        raise ValueError("Язык не указан в прогрессе! Укажите 'language' в reset_session_counters.")

    
    progress["session"]["topic_offers"] = 0
    progress["session"]["topic_offers_rejected"] = 0
    progress["session"]["words_learned_this_session"] = []
    progress["language"] = lang

    save_progress(progress, lang)
    print(f"🔄 Счётчики сессии [{lang}] обнулены")


def get_session_time(progress: dict) -> tuple[int, bool]:
    """Возвращает (минуты с начала сессии, нужно ли обнулить счётчики)."""
    session = progress.get("session", {})
    start_time_str = session.get("start_time")
    
    if not start_time_str:
        return 0, True  # Нет сессии → нужно начать новую
    
    start_time = datetime.fromisoformat(start_time_str)
    elapsed = datetime.now() - start_time
    
    # Если прошло больше 2 часов без активности — начинаем новую сессию
    if elapsed.total_seconds() > 7200:  # 2 часа
        return int(elapsed.total_seconds() // 60), True
    
    return int(elapsed.total_seconds() // 60), False


def get_session_language(progress: dict) -> str:
    """Возвращает язык текущей сессии.
    
    Args:
        progress: Словарь с прогрессом
        
    Returns:
        Язык сессии ('en' или 'ru')
    """
    return progress.get("language", "ru")


def save_progress(progress: dict, lang: str = 'en'):
    """Сохраняет прогресс в файл.
    
    Args:
        progress: Словарь с прогрессом (должен содержать поле 'language')
        
    Raises:
        ValueError: Если язык не указан в прогрессе
    """
    if not lang:
        raise ValueError("Язык не указан в прогрессе! Укажите 'language' в словаре.")
    
    print(f"💾 Сохранение прогресса [{lang}] в: {EN_PROGRESS_FILE if lang == 'en' else RU_PROGRESS_FILE}")
    print(f"💾 Слов для сохранения: {len(progress.get('learned_words', {}))}")
    
    try:
        # Создаём папку, если её нет
        progress_file = EN_PROGRESS_FILE if lang == "en" else RU_PROGRESS_FILE
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Прогресс [{lang}] сохранён! Размер: {progress_file.stat().st_size} байт")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения [{lang}]: {e}")


def update_word_progress(progress: dict, word: str, success: bool = True, level: str = None):
    """Обновляет прогресс по конкретному слову.
    
    Args:
        word: Слово для обновления
        success: Успех (True) или ошибка (False)
        lang: Язык прогресса (опционально). Если None — определяется из транскрипции.
        
    Returns:
        Обновлённый словарь прогресса
    """
    print(f"📝 Обновление слова: '{word}' (success={success})")
    
    word = word.lower().strip()

    if word not in progress["learned_words"]:
        progress["learned_words"][word] = {
            "correct_count": 0,
            "wrong_count": 0,
            "last_practiced": None,
            "first_learned": None
        }
    
    if success:
        progress["learned_words"][word]["correct_count"] += 1
        if not progress["learned_words"][word]["first_learned"]:
            progress["learned_words"][word]["first_learned"] = datetime.now().isoformat()
    else:
        progress["learned_words"][word]["correct_count"] -= 2
        
    progress["learned_words"][word]["last_practiced"] = datetime.now().isoformat()

    if level:
        progress["learned_words"][word]["level"] = level

    return progress


def get_learned_words(progress: dict = None, lang: str = None) -> list:
    """Возвращает список выученных слов (правильно ответил 2+ раз).
    
    Args:
        progress: Словарь с прогрессом. Если None — загружается.
        lang: Язык прогресса. Если None — определяется из прогресса.
        
    Returns:
        Возвращает список выученных слов, отсортированный по дате последнего использования (сначала недавние).
    """
    if progress is None:
        progress = load_progress(lang)
    
    # Определяем язык, если не указан в прогрессе
    if lang is None:
        lang = progress.get("language", "en")

    THRESHOLDS = {
        "A1": 3,
        "A2": 8,
        "B1": 14,
    }
    
    words_data = []
    for word, data in progress["learned_words"].items():
        correct_count = data.get("correct_count", 0)
        level = data.get("level", 'A1')
        threshold = THRESHOLDS.get(level, 1)
        if correct_count >= threshold:
            last_practiced = data.get("last_practiced")
            words_data.append({
                "word": word,
                "level": level,
                "threshold": threshold,
                "last_practiced": last_practiced
            })

    # None → отправляем в конец
    words_data.sort(
        key=lambda x: (x["last_practiced"] is None, x["last_practiced"] if x["last_practiced"] else ""),
        reverse=True
    )

    return words_data


def get_words_needing_review(progress: dict = None, lang: str = None, days: int = 7) -> list:
    """Возвращает слова, которые нужно повторить (не практиковались более N дней).
    
    Args:
        progress: Словарь с прогрессом. Если None — загружается.
        lang: Язык прогресса. Если None — определяется из прогресса.
        days: Количество дней без практики
        
    Returns:
        Список слов для повторения
    """
    if progress is None:
        progress = load_progress(lang)
    
    # Определяем язык, если не указан в прогрессе
    if lang is None:
        lang = progress.get("language", "ru")
    
    from datetime import datetime, timedelta
    now = datetime.now()
    threshold = now - timedelta(days=days)
    
    review_needed = []
    for word, data in progress["learned_words"].items():
        last = data.get("last_practiced")
        if last:
            last_date = datetime.fromisoformat(last)
            if last_date < threshold:
                review_needed.append(word)
        else:
            review_needed.append(word)
    
    return review_needed