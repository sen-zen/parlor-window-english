import re


def is_english_word(word: str, english_dict: set[str] = None) -> bool:
    """Проверяет, является ли слово английским.
    
    Поддерживает одиночные слова и фразы с пробелами.
    Служебные слова фильтруются только если это отдельное слово (не часть фразы).
    
    Args:
        word: Слово или фраза для проверки
        english_dict: Опциональный словарь английских слов для точной проверки
    
    Returns:
        True если слово/фраза английское, False иначе
    """
    if not word or len(word.strip()) < 2:
        return False
    
    # Разрешаем буквы, пробелы, дефисы и апострофы
    if not re.match(r'^[a-zA-Z\s\'\-]+$', word):
        return False
    
    word = word.strip()
    
    if len(word) < 2 or len(word) > 100:
        return False
    
    # Разбиваем на слова для проверки служебных слов
    words = word.lower().split()
    
    # Служебные слова (пропускаем только отдельные, не в фразе)
    skip_words = {'i', 'me', 'my', 'you', 'he', 'she', 'it', 'we', 'they', 'them',
                  'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                  'the', 'a', 'an', 'this', 'that', 'these', 'those',
                  'and', 'or', 'but', 'so', 'for', 'nor', 'yet',
                  'to', 'from', 'with', 'without', 'at', 'by', 'on', 'in', 'of', 'about'}
    
    # Если фраза (есть пробелы) — служебные слова не фильтруем
    if ' ' in word:
        return True
    
    # Для одиночного слова проверяем, что оно не служебное
    if word.lower() in skip_words:
        return False
    
    return True