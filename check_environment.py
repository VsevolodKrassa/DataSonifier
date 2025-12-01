#!/usr/bin/env python3
"""
Скрипт для проверки окружения DataSonifier
"""

import sys

REQUIRED_VERSIONS = {
    'numpy': '1.21.6',
    'scipy': '1.7.3', 
    'matplotlib': '3.5.3',
    'soundfile': '0.12.1'
}

def check_version(actual, required, name):
    """Проверяет соответствие версии"""
    try:
        actual_tuple = tuple(map(int, actual.split('.')[:3]))
        required_tuple = tuple(map(int, required.split('.')[:3]))
        if actual_tuple == required_tuple:
            return True, f"✅ {name}: {actual} - OK"
        else:
            return False, f"❌ {name}: {actual} != {required} (требуется)"
    except Exception as e:
        return False, f"❌ {name}: ошибка проверки версии - {e}"

def check_environment():
    print("🔍 Проверка окружения DataSonifier...")
    print()
    
    # Проверка Python
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python: {python_version}")
    if sys.version_info >= (3, 7):
        print("✅ Python 3.7+ - OK")
    else:
        print("❌ Требуется Python 3.7+")
        return False
    
    print()
    
    # Проверка библиотек
    all_ok = True
    for package, required_version in REQUIRED_VERSIONS.items():
        try:
            module = __import__(package)
            actual_version = getattr(module, '__version__', 'unknown')
            
            is_ok, message = check_version(actual_version, required_version, package)
            print(message)
            if not is_ok:
                all_ok = False
                
        except ImportError:
            print(f"❌ {package}: НЕ УСТАНОВЛЕН")
            all_ok = False
    
    print()
    if all_ok:
        print("✅ Окружение настроено правильно!")
        print("Запустите: python datasonifier.py путь/к/файлу.txt")
    else:
        print("❌ Окружение не соответствует требованиям")
        print("⚠️ Установите точные версии: pip install -r requirements.txt")
    
    return all_ok

if __name__ == "__main__":
    check_environment()
