"""Тестирование различных вариантов JSON-ответов от Ollama."""

import json
from utils.json_parser import safe_json_parse


def test_case(name: str, text: str, expected_success: bool = True):
    """Запуск теста с выводом результата."""
    print(f"\n{'='*60}")
    print(f"Тест: {name}")
    print(f"{'='*60}")
    print(f"Входной текст:")
    print(text[:200] + "..." if len(text) > 200 else text)
    
    try:
        result = safe_json_parse(text)
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
    
    status = "✅ УСПЕХ" if success else "❌ ОШИБКА"
    print(f"\n{status}")
    if not expected_success and success:
        print("⚠️  Ожидался провал, но парсинг прошёл!")
    elif not success:
        print(f"Ошибка: {error}")
    
    if success:
        print(f"\nРаспарсенный результат:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Проверка обязательных полей
        if "text" in result:
            print(f"✅ Поле 'text' найдено: {result['text'][:50]}...")
        else:
            print("❌ Поле 'text' НЕ найдено!")
        
        if "words" in result:
            print(f"✅ Поле 'words' найдено: {result['words']}")
        else:
            print("⚠️  Поле 'words' отсутствует (опционально)")
        
        if "mistakes" in result:
            print(f"✅ Поле 'mistakes' найдено: {result['mistakes']}")
        else:
            print("⚠️  Поле 'mistakes' отсутствует (опционально)")
    
    return success


# ============ ВАРИАНТ 1: Чистый JSON ============
test_case(
    "Вариант 1: Чистый JSON",
    '{"text": "Отлично! Ты правильно сказал Hello. Как будет Goodbye?", "words": ["hello"], "topic": "Приветствия"}'
)

# ============ ВАРИАНТ 2: JSON с экранированными кавычками ============
test_case(
    "Вариант 2: JSON с экранированными кавычками",
    '{"text": "Отлично! Ты правильно сказал \\"Hello\\". Как будет \\"Goodbye\\"?", "words": ["hello"], "topic": "Приветствия"}'
)

# ============ ВАРИАНТ 3: JSON с преамбулой от модели ============
test_case(
    "Вариант 3: JSON с преамбулой",
    'Конечно! Вот твой ответ:\n{"text": "Отлично! Ты правильно сказал Hello. Как будет Goodbye?", "words": ["hello"], "topic": "Приветствия"}\nЭто конец.'
)

# ============ ВАРИАНТ 4: JSON с thinking (CoT) ============
test_case(
    "Вариант 4: JSON внутри thinking",
    'thinking: {"text": "Отлично! Ты правильно сказал Hello. Как будет Goodbye?", "words": ["hello"], "topic": "Приветствия"}\n\ncontent: Привет!'
)

# ============ ВАРИАНТ 5: JSON с русскими кавычками ============
test_case(
    "Вариант 5: JSON с русскими кавычками",
    '{"text": "Отлично! Ты правильно сказал «Hello». Как будет «Goodbye»?", "words": ["hello"], "topic": "Приветствия"}'
)

# ============ ВАРИАНТ 6: JSON без поля text ============
test_case(
    "Вариант 6: JSON без поля text",
    '{"words": ["hello"], "mistakes": [], "topic": "Приветствия"}',
    expected_success=False
)

# ============ ВАРИАНТ 7: JSON с пустым текстом ============
test_case(
    "Вариант 7: JSON с пустым текстом",
    '{"text": "", "words": [], "mistakes": []}'
)

# ============ ВАРИАНТ 8: JSON с пробелами и переносами строк ============
test_case(
    "Вариант 8: JSON с пробелами и переносами",
    '''{
        "text": "Отлично! Ты правильно сказал Hello. Как будет Goodbye?",
        "words": ["hello"],
        "topic": "Приветствия"
    }'''
)

# ============ ВАРИАНТ 9: JSON с вложенными объектами (неподдерживается) ============
test_case(
    "Вариант 9: JSON с вложенными объектами",
    '{"text": "Привет!", "data": {"nested": {"value": 42}}}'
)

# ============ ВАРИАНТ 10: JSON с массивом строк ============
test_case(
    "Вариант 10: JSON с массивом",
    '{"text": "Привет!", "words": ["hello", "world", "foo"], "mistakes": ["bred"]}'
)

# ============ ВАРИАНТ 11: Текст без JSON (ошибка) ============
test_case(
    "Вариант 11: Текст без JSON",
    'Привет! Как дела?',
    expected_success=False
)

# ============ ВАРИАНТ 12: JSON с Unicode символами ============
test_case(
    "Вариант 12: JSON с Unicode",
    '{"text": "Отлично! Ты сказал Привет (Hello)! Это значит Welcome.", "words": ["hello"], "topic": "Приветствия"}'
)

# ============ ВАРИАНТ 13: JSON с цифрами в тексте ============
test_case(
    "Вариант 13: JSON с цифрами",
    '{"text": "Ты сказал Hello! Это число 42. Сколько будет 2+2?", "words": ["hello"], "topic": "Математика"}'
)

# ============ ВАРИАНТ 14: JSON с булевыми значениями (неподдерживается) ============
test_case(
    "Вариант 14: JSON с булевыми значениями",
    '{"text": "Привет!", "active": true, "count": 5}'
)

# ============ ВАРИАНТ 15: JSON с null значениями (неподдерживается) ============
test_case(
    "Вариант 15: JSON с null",
    '{"text": "Привет!", "optional_field": null}'
)

print("\n" + "="*60)
print("Тестирование завершено!")
print("="*60)