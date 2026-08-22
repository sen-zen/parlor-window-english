from typing import Any
import json
import re


def extract_json_object(text: str) -> str:
    """
    Извлекает JSON-объект из текста, используя баланс скобок.
    
    Args:
        text: Текст, содержащий JSON
        
    Returns:
        Строка с JSON-объектом или пустую строку если не найдено
    """
    # Ищем начало объекта
    start = text.find('{')
    if start == -1:
        return ""
    
    # Балансирование скобок с учётом экранирования
    balance = 0
    end = start
    in_string = False
    escape = False
    
    for i, char in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0:
                    end = i + 1
                    break
    
    if end == start:
        return ""
    
    return text[start:end]


def find_json_key_value_pairs(json_str: str) -> list[tuple[str, str]]:
    """
    Находит все пары ключ-значение в JSON-объекте.
    
    Args:
        json_str: Строка с JSON-объектом
        
    Returns:
        Список кортежей (ключ, значение)
    """
    pairs = []
    
    # Регулярка для поиска пар ключ-значение
    # Ключ в кавычках, затем : и значение (строка, число, булево, null или объект/массив)
    pattern = r'"([^"]+)"\s*:\s*(?:(?:[^"\\]|\\.)+(?:"(?:[^"\\]|\\.)*"\s*:\s*)?(?:[^,"{}\[\]\\]|\\.|")|(?:"(?:[^"\\]|\\.)*"))'
    
    for match in re.finditer(pattern, json_str, re.DOTALL):
        key = match.group(1)
        value = match.group(2).strip()
        pairs.append((key, value))
    
    return pairs


def parse_json_value(value_str: str) -> Any:
    """
    Парсит значение любого типа из строки.
    
    Args:
        value_str: Строка с JSON-значением
        
    Returns:
        Значение соответствующего типа (str, int, float, bool, list, dict, None)
    """
    value_str = value_str.strip()
    
    # Пустое значение
    if not value_str:
        return ""
    
    # Строка
    if value_str.startswith('"'):
        # Извлекаем строку с учётом экранирования
        match = re.match(r'"((?:[^"\\]|\\.)*)"', value_str)
        if match:
            return match.group(1).replace('\\"', '"').replace('\\\\', '\\')
        return value_str[1:-1]  # Убираем кавычки
    
    # Число (целое или плавающее)
    if re.match(r'^-?\d+\.?\d*$', value_str):
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    
    # Булево
    if value_str == 'true':
        return True
    if value_str == 'false':
        return False
    
    # Null
    if value_str == 'null':
        return None
    
    # Массив [...]
    if value_str.startswith('['):
        items = re.findall(r'\[(?:[^"\\]|\\.)+(?:"(?:[^"\\]|\\.)*"\s*:\s*)?(?:[^,"{}\[\]\\]|\\.|")?\s*,?\]', value_str)
        parsed_items = []
        for item in items:
            item = item.strip()
            if item.startswith('"'):
                match = re.match(r'"((?:[^"\\]|\\.)*)"', item)
                if match:
                    parsed_items.append(match.group(1).replace('\\"', '"').replace('\\\\', '\\'))
                else:
                    parsed_items.append(item[1:-1])
            elif item:
                parsed_items.append(parse_json_value(item))
        return parsed_items
    
    # Объект {...} — оставляем как строку для дальнейшего анализа
    if value_str.startswith('{'):
        return value_str
    
    # Возвращаем строку как есть
    return value_str


def safe_json_parse(text: str) -> dict:
    """
    Универсальный парсер JSON.
    Извлекает JSON из любого текста и возвращает все поля.
    """
    # 1. Находим JSON-объект
    start = text.find('{')
    if start == -1:
        return {}
    
    balance = 0
    end = start
    in_string = False
    escape = False
    
    for i, char in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0:
                    end = i + 1
                    break
    
    if end == start:
        return {}
    
    json_str = text[start:end]
    
    # 2. Пробуем стандартный парсинг
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # 3. Универсальное извлечение всех ключей
    result = {}
    
    # Находим все пары "ключ": "значение"
    # Ключи могут быть любыми: text, words, mistakes, topic, и т.д.
    pattern = r'"([^"]+)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|\[([^\]]*)\]|(\{[^}]*\})|([^,}]+))'
    
    for match in re.finditer(pattern, json_str, re.DOTALL):
        key = match.group(1)
        
        # Строка
        if match.group(2) is not None:
            value = match.group(2)
            # Очищаем экранирование
            value = value.replace('\\"', '"').replace('\\\\', '\\')
            result[key] = value
        # Массив
        elif match.group(3) is not None:
            arr_str = match.group(3)
            items = re.findall(r'"([^"]*)"', arr_str)
            result[key] = items
        # Вложенный объект
        elif match.group(4) is not None:
            obj_str = match.group(4)
            try:
                result[key] = json.loads(obj_str)
            except:
                result[key] = obj_str
        # Простое значение (число, булево, null)
        elif match.group(5) is not None:
            val_str = match.group(5).strip()
            if val_str.isdigit():
                result[key] = int(val_str)
            elif val_str.replace('.', '').isdigit():
                result[key] = float(val_str)
            elif val_str == 'true':
                result[key] = True
            elif val_str == 'false':
                result[key] = False
            elif val_str == 'null':
                result[key] = None
            else:
                result[key] = val_str
    
    return result
