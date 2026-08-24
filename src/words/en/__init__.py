"""Словари английского языка по уровням."""

from .a1 import A1_WORDS
from .a2 import A2_WORDS
from .b1 import B1_WORDS

__all__ = ["A1_WORDS", "A2_WORDS", "B1_WORDS"]

# Словари по уровням
LEVELS = {
    "A1": A1_WORDS,
    "A2": A2_WORDS,
    "B1": B1_WORDS,
}

def get_words_by_level(level: str) -> list:
    """Возвращает список слов для указанного уровня."""
    return LEVELS.get(level.upper(), [])

def get_word_level(word: str) -> str:
    """Определяет уровень слова на основе списков."""
    word = word.lower().strip()
    if word in A1_WORDS:
        return "A1"
    elif word in A2_WORDS:
        return "A2"
    elif word in B1_WORDS:
        return "B1"
    else:
        return "A1"

def get_all_words() -> list:
    """Возвращает все слова (3000)."""
    return B1_WORDS
