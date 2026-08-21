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
    Универсальный парсер JSON из текста.
    
    Автоматически определяет:
    - Границы JSON-объекта (баланс скобок)
    - Все ключи и их значения любого типа
    - Вложенные объекты и массивы
    
    Args:
        text: Текст, содержащий JSON
        
    Returns:
        Словарь с распарсенными данными
    """
    # 1. Извлекаем JSON-объект
    json_str = extract_json_object(text)
    
    if not json_str:
        return {}
    
    # 2. Пробуем стандартный парсинг
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # 3. Извлекаем пары ключ-значение через регулярки
    result = {}
    pairs = find_json_key_value_pairs(json_str)
    
    for key, value_str in pairs:
        parsed_value = parse_json_value(value_str)
        
        # Для объектов и массивов — рекурсивный парсинг
        if isinstance(parsed_value, str):
            if parsed_value.startswith('{'):
                try:
                    nested_obj = json.loads('{' + parsed_value + '}')
                    result[key] = nested_obj
                except:
                    result[key] = parsed_value
            elif parsed_value.startswith('['):
                try:
                    nested_arr = json.loads('[' + parsed_value + ']')
                    result[key] = nested_arr
                except:
                    result[key] = parsed_value
        else:
            result[key] = parsed_value
    
    return result