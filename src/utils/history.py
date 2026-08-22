"""Модуль для работы с историей диалогов."""

import json
from pathlib import Path
from typing import List, Dict, Any


# Пути к файлам истории
EN_HISTORY_FILE = Path(__file__).parent.parent / "stores" / "en.history.json"
RU_HISTORY_FILE = Path(__file__).parent.parent / "stores" / "ru.history.json"

MAX_HISTORY_MESSAGES = 20  # Максимум сообщений в истории
MAX_HISTORY_CHARS = 8000   # Примерный лимит символов


def trim_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Обрезает историю до лимитов.
    
    Сохраняет системные сообщения, обрезает остальные.
    
    Args:
        history: Список сообщений истории
        
    Returns:
        Обрезанный список сообщений
    """
    if not history:
        return []
    
    # 1. Обрезка по количеству сообщений
    if len(history) > MAX_HISTORY_MESSAGES:
        # Сохраняем системные сообщения
        system_msgs = [m for m in history if m.get("role") == "system"]
        other_msgs = [m for m in history if m.get("role") != "system"]
        
        # Оставляем последние N сообщений
        other_msgs = other_msgs[-MAX_HISTORY_MESSAGES:]
        history = system_msgs + other_msgs
        print(f"📋 История обрезана по количеству: {len(history)} сообщений")
    
    # 2. Обрезка по символам (дополнительно)
    total_chars = sum(len(str(m.get("content", ""))) for m in history)
    if total_chars > MAX_HISTORY_CHARS:
        # Удаляем старые сообщения, пока не уложимся в лимит
        # Сохраняем минимум 3 сообщения (системное + последние 2)
        while total_chars > MAX_HISTORY_CHARS and len(history) > 3:
            # Удаляем второе сообщение (первое обычно системное)
            removed = history.pop(1)
            total_chars -= len(str(removed.get("content", "")))
            print(f"📋 Удалено старое сообщение для экономии места")
    
    return history


def load_history(lang: str = "en") -> List[Dict[str, Any]]:
    """Загружает историю диалогов для указанного языка.
    
    Args:
        lang: Язык ("en" или "ru")
        
    Returns:
        Список сообщений истории
    """
    history_file = EN_HISTORY_FILE if lang == "en" else RU_HISTORY_FILE
    
    if not history_file.exists():
        return []
    
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
            return trim_history(history)
    except (json.JSONDecodeError, IOError):
        return []


def save_history(history: List[Dict[str, Any]], lang: str = "en") -> None:
    """Сохраняет историю диалогов для указанного языка.
    
    Args:
        history: Список сообщений истории
        lang: Язык ("en" или "ru")
    """
    history_file = EN_HISTORY_FILE if lang == "en" else RU_HISTORY_FILE
    trimmed_history = trim_history(history)

    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(trimmed_history, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Ошибка сохранения истории: {e}")


def append_history(message: Dict[str, Any], lang: str = "en") -> None:
    """Добавляет сообщение в историю диалогов.
    
    Args:
        message: Сообщение для добавления
        lang: Язык ("en" или "ru")
    """
    history = load_history(lang)
    history.append(message)
    save_history(history, lang)


def get_history(limit: int = None, lang: str = "en") -> List[Dict[str, Any]]:
    """Получает историю диалогов с ограничением по количеству сообщений.
    
    Args:
        limit: Максимальное количество сообщений (None для всех)
        lang: Язык ("en" или "ru")
        
    Returns:
        Список последних сообщений истории
    """
    history = load_history(lang)
    
    if limit is not None:
        history = history[-limit:]
    
    return history


def clear_history(lang: str = "en") -> None:
    """Очищает историю диалогов для указанного языка.
    
    Args:
        lang: Язык ("en" или "ru")
    """
    save_history([], lang)


# Функции для работы с историей по темам
def get_topic_history(topic: str, limit: int = 50, lang: str = "en") -> List[Dict[str, Any]]:
    """Получает историю сообщений по теме.
    
    Args:
        topic: Тема урока
        limit: Максимальное количество сообщений
        lang: Язык ("en" или "ru")
        
    Returns:
        Список сообщений истории по теме
    """
    history = get_history(limit, lang)
    
    # Фильтруем сообщения по теме (простая проверка в тексте)
    topic_history = []
    for msg in history:
        content = msg.get("content", "").lower()
        if topic.lower() in content or topic.lower() in msg.get("text", "").lower():
            topic_history.append(msg)
    
    return topic_history


def get_last_message(lang: str = "en") -> Dict[str, Any]:
    """Получает последнее сообщение из истории.
    
    Args:
        lang: Язык ("en" или "ru")
        
    Returns:
        Последнее сообщение или пустой словарь
    """
    history = load_history(lang)
    return history[-1] if history else {}


def get_last_assistant_message(lang: str = "en") -> Dict[str, Any]:
    """Получает последнее сообщение от ассистента.
    
    Args:
        lang: Язык ("en" или "ru")
        
    Returns:
        Последнее сообщение ассистента или пустой словарь
    """
    history = load_history(lang)
    
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return msg
    
    return {}


def get_last_user_message(lang: str = "en") -> Dict[str, Any]:
    """Получает последнее сообщение пользователя.
    
    Args:
        lang: Язык ("en" или "ru")
        
    Returns:
        Последнее сообщение пользователя или пустой словарь
    """
    history = load_history(lang)
    
    for msg in reversed(history):
        if msg.get("role") == "user":
            return msg
    
    return {}


def get_conversation_summary(lang: str = "en") -> Dict[str, Any]:
    """Получает краткую сводку разговора.
    
    Args:
        lang: Язык ("en" или "ru")
        
    Returns:
        Словарь со сводкой: количество сообщений, тем, слов и т.д.
    """
    history = load_history(lang)
    
    if not history:
        return {
            "message_count": 0,
            "topics": [],
            "learned_words": [],
            "mistakes": [],
        }
    
    # Собираем уникальные темы
    topics = set()
    for msg in history:
        if "topic" in msg:
            topics.add(msg["topic"])
    
    # Собираем выученные слова и ошибки
    learned_words = []
    mistakes = []
    for msg in history:
        if isinstance(msg.get("content"), dict):
            words = msg["content"].get("words", [])
            mistakes_list = msg["content"].get("mistakes", [])
            learned_words.extend(words)
            mistakes.extend(mistakes_list)
    
    return {
        "message_count": len(history),
        "topics": list(topics),
        "learned_words": list(set(learned_words)),
        "mistakes": list(set(mistakes)),
    }