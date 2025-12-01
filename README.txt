Инструкция по установке и запуску DataSonifier

О программе:

	DataSonifier - это инструмент для сонификации научных данных

	Преобразуйте данные из PowerGraph в звук - слушайте ваши эксперименты!

	Open Source решение от art&science исследовательской группы KVEF

Установка программы:

1. Установите Python 3.10
Скачайте с официального сайта: https://www.python.org/downloads/release/python-31011/

Важно при установке:

	✅ Отметьте галочку "Add Python to PATH" (это критически важно!)

	✅ Выберите опцию "Install Now"

2. Установите зависимостей

Windows:

	Откройте командную строку (Win + R → введите cmd → Enter)

	Перейдите в папку с программой:

	cmd
		cd C:\ПУТЬ\К\ПАПКЕ\datasonifier

	Установите зависимости:

	cmd
		pip install -r requirements.txt

macOS:

	Откройте Терминал (Finder → Программы → Утилиты → Терминал)

	Перейдите в папку с программой:

	bash
		cd /ПУТЬ/К/ПАПКЕ/datasonifier
	Установите зависимости:

	bash
		pip3 install -r requirements.txt
Linux:

	Откройте терминал

	Перейдите в папку с программой:

	bash
		cd /ПУТЬ/К/ПАПКЕ/datasonifier
	Установите зависимости:

	bash
		pip3 install -r requirements.txt

3. Проверка установки

Windows:

	cmd
		python check_environment.py

macOS/Linux:

	bash
		python3 check_environment.py

✅ Должно появиться: "Окружение настроено правильно!"

Запуск программы

Способ 1: Указать файл при запуске (рекомендуется)

Windows:

	cmd
		cd C:\ПУТЬ\К\ПАПКЕ\datasonifier
		py -3.10 datasonifier.py C:\ПУТЬ\К\ФАЙЛУ\данные.txt

macOS/Linux:

	bash
		cd /ПУТЬ/К/ПАПКЕ/datasonifier
		python3 datasonifier.py /ПУТЬ/К/ФАЙЛУ/данные.txt

Способ 2: Интерактивный ввод файла

Windows:

	cmd
		cd C:\ПУТЬ\К\ПАПКЕ\datasonifier
		py -3.10 datasonifier.py

	Затем введите путь к файлу когда программа запросит

macOS/Linux:

	bash
		cd /ПУТЬ/К/ПАПКЕ/datasonifier
		python3 datasonifier.py

	Затем введите путь к файлу когда программа запросит

Формат входного файла

Программа ожидает текстовый файл в формате:

text
Rate:	1000
Step:	0.001
Duration:	10.0
Size:	10000
Time, s	Data, V
0.000	0.124
0.001	0.135
0.002	0.128
...


Примеры реальных путей

Windows:

	C:\Users\Ivanov\Desktop\datasonifier\experiment_data.txt

	D:\Research\data_measurements.txt

macOS:

	/Users/annasmith/Documents/datasonifier/experiment_data.txt

	/Desktop/lab_measurements.txt

Linux:

	/home/user/datasonifier/experiment_data.txt

	/home/user/Downloads/lab_data.txt

Решение частых проблем

❌ Если команда python не работает:

	Windows: Используйте py -3.10 вместо python
	macOS/Linux: Используйте python3 вместо python

❌ Если команда pip не работает:

	Windows: Используйте py -3.10 -m pip
	macOS/Linux: Используйте python3 -m pip

❌ Если файл не найден:

	Убедитесь, что путь указан правильно

Используйте двойные кавычки если в пути есть пробелы:

	cmd
		py -3.10 datasonifier.py "C:\My Documents\data file.txt"

❌ Если графики не показываются:

	cmd
		pip install tk


Быстрый запуск (для Windows)

Создайте файл запуск.bat в папке с программой:

	bat
		@echo off
		cd /d "C:\ПУТЬ\К\ПАПКЕ\datasonifier"
		py -3.10 datasonifier.py
		pause

	Теперь просто запускайте двойным кликом по этому файлу!

Что делать после запуска

Программа покажет график ваших данных

Настройте параметры преобразования (частоты, пороги)

Программа создаст WAV-файл с преобразованным звуком

Выберите вариант визуализации результатов

