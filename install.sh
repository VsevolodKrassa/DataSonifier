#!/bin/bash
echo "Установка DataSonifier..."
echo

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не установлен"
    echo "💡 Установите:"
    echo "   Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "   macOS: brew install python"
    echo "   или скачайте с python.org"
    exit 1
fi

# Проверка версии Python
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.7"

if [ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
    echo "✅ Python $PYTHON_VERSION - OK"
else
    echo "❌ Python $PYTHON_VERSION < $REQUIRED_VERSION (требуется 3.7+)"
    exit 1
fi

echo
echo "📦 Устанавливаю зависимости..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo
    echo "✅ Установка завершена!"
    echo
    echo "🔍 Проверяю окружение..."
    python3 check_environment.py
    
    echo
    echo "🚀 Запуск программы:"
    echo "   python3 datasonifier.py путь/к/файлу.txt"
else
    echo
    echo "❌ Ошибка установки зависимостей"
    echo "💡 Попробуйте: pip3 install --upgrade pip"
    exit 1
fi