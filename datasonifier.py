#!/usr/bin/env python3
"""
DataSonifier - программа для сонификации данных
Разработана art&science группой KVEF
"""

import sys
import os
import numpy as np
from scipy import signal
from scipy.signal import spectrogram, firwin, lfilter
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

def check_version(actual, required, name):
    """Проверяет соответствие версии"""
    try:
        actual_tuple = tuple(map(int, actual.split('.')[:3]))
        required_tuple = tuple(map(int, required.split('.')[:3]))
        if actual_tuple != required_tuple:
            print(f"❌ {name} {actual} != {required} (требуется точное соответствие)")
            return False
        return True
    except Exception as e:
        print(f"❌ Ошибка проверки версии {name}: {e}")
        return False

def check_environment():
    """Проверяет все зависимости"""
    if sys.version_info < (3, 7):
        print("❌ DataSonifier требует Python 3.7 или выше!")
        print(f"💡 Текущая версия: {sys.version}")
        return False

    try:
        import numpy as np
        if not check_version(np.__version__, "1.21.6", "NumPy"):
            return False
    except ImportError:
        print("❌ NumPy не установлен!")
        return False

    try:
        import scipy
        if not check_version(scipy.__version__, "1.7.3", "SciPy"):
            return False
    except ImportError:
        print("❌ SciPy не установлен!")
        return False

    try:
        import matplotlib
        if not check_version(matplotlib.__version__, "3.5.3", "Matplotlib"):
            return False
    except ImportError:
        print("❌ Matplotlib не установлен!")
        return False

    try:
        import soundfile as sf
        if not check_version(sf.__version__, "0.12.1", "SoundFile"):
            return False
    except ImportError:
        print("❌ SoundFile не установлен!")
        return False

    return True

class DataSonifier:
    def __init__(self):
        self.metadata = {}
        self.raw_data = None
        self.processed_data = None
        self.audio_data = None

    def print_banner(self):
        banner = """
        ╔═══════════════════════════════════════╗
        ║          DataSonifier v1.0            ║
        ║    Преобразование данных в звук       ║
        ║                                       ║
        ║        Open Source by KVEF            ║
        ║    art&science research group         ║
        ╚═══════════════════════════════════════╝
        """
        print(banner)
        print("📝 Использование: python datasonifier.py [путь/к/файлу.txt]")

    def load_file(self, filename):
        print(f"📁 Загружаю файл: {filename}")
        
        if not os.path.exists(filename):
            print(f"❌ Файл не найден!")
            return False
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.metadata, data_lines = self._parse_metadata(lines)
            self.raw_data = self._parse_data(data_lines)
            
            if len(self.raw_data) == 0:
                print("❌ Нет данных для обработки!")
                return False
            
            print(f"✅ Загружено {len(self.raw_data):,} точек")
            if 'Rate' in self.metadata:
                duration = len(self.raw_data) / self.metadata['Rate']
                print(f"   • Длительность: {duration:.2f} сек")
            
            return True
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False

    def _parse_metadata(self, lines):
        metadata = {}
        data_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Rate:'):
                metadata['Rate'] = int(line.split('\t')[1])
            elif line.startswith('Step:'):
                metadata['Step'] = float(line.split('\t')[1])
            elif line.startswith('Duration:'):
                metadata['Duration'] = float(line.split('\t')[1])
            elif line.startswith('Size:'):
                metadata['Size'] = int(line.split('\t')[1])
            elif line.startswith('Time, s') or self._is_data_line(line):
                data_start = i
                if line.startswith('Time, s'):
                    data_start += 1
                break
        
        if 'Rate' not in metadata and 'Step' in metadata:
            metadata['Rate'] = int(1.0 / metadata['Step'])
        
        return metadata, lines[data_start:]

    def _is_data_line(self, line):
        parts = line.split('\t')
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
                return True
            except ValueError:
                pass
        return False

    def _parse_data(self, data_lines):
        data = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    data.append(float(parts[1]))
                except ValueError:
                    continue
        return np.array(data)

    def analyze_data(self):
        if self.raw_data is None:
            print("❌ Данные не загружены!")
            return False

        data_min = np.min(self.raw_data)
        data_max = np.max(self.raw_data)

        if data_max - data_min > 0:
            normalized_data = (self.raw_data - data_min) / (data_max - data_min)
            suggested_lower = np.percentile(normalized_data, 10)
            suggested_upper = np.percentile(normalized_data, 90)
        else:
            suggested_lower = 0.1
            suggested_upper = 0.9

        stats = {
            'min': data_min,
            'max': data_max,
            'mean': np.mean(self.raw_data),
            'std': np.std(self.raw_data),
            'suggested_lower_threshold': suggested_lower,
            'suggested_upper_threshold': suggested_upper
        }

        print("\n📊 Анализ данных:")
        print(f"   • Минимальное значение: {stats['min']:.4f} V")
        print(f"   • Максимальное значение: {stats['max']:.4f} V")
        print(f"   • Среднее: {stats['mean']:.4f} V")
        print(f"   • Стандартное отклонение: {stats['std']:.4f} V")
        print(f"   • Предложенный нижний порог: {stats['suggested_lower_threshold']:.3f}")
        print(f"   • Предложенный верхний порог: {stats['suggested_upper_threshold']:.3f}")
        print(f"   • Динамический диапазон: {stats['max'] - stats['min']:.4f} V")
        print(f"   • Пороги задаются в диапазоне [0,1] после нормализации")

        return stats

    def plot_raw_data(self):
        if self.raw_data is None:
            print("❌ Данные не загружены!")
            return False
        
        sample_rate = self.metadata.get('Rate', 1000)
        time_axis = np.arange(len(self.raw_data)) / sample_rate
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(time_axis, self.raw_data, alpha=0.8, linewidth=0.5, color='blue')
        ax1.set_title('Исходные данные')
        ax1.set_ylabel('Напряжение (В)')
        ax1.grid(True, alpha=0.3)
        
        data_min = np.min(self.raw_data)
        data_max = np.max(self.raw_data)
        
        if data_max - data_min > 0:
            normalized_data = (self.raw_data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(self.raw_data)
        
        ax2.plot(time_axis, normalized_data, alpha=0.8, linewidth=0.5, color='green')
        ax2.set_title('Нормализованные данные')
        ax2.set_xlabel('Время (секунды)')
        ax2.set_ylabel('Нормализованное значение [0,1]')
        ax2.grid(True, alpha=0.3)
        
        ax2.text(0.02, 0.98, f'Диапазон: [{data_min:.4f}, {data_max:.4f}] В → [0, 1]', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        return True

    def get_processing_parameters(self, stats):
        print("\n🎛  НАСТРОЙКА ПАРАМЕТРОВ")
        print("   Пороги задаются в диапазоне [0,1]")
        
        lower_threshold_input = input(
            f"Нижний порог [0-1] [рекомендуется {stats['suggested_lower_threshold']:.3f}]: "
        ).strip()
        lower_threshold = float(lower_threshold_input) if lower_threshold_input else stats['suggested_lower_threshold']
        
        upper_threshold_input = input(
            f"Верхний порог [0-1] [рекомендуется {stats['suggested_upper_threshold']:.3f}]: "
        ).strip()
        upper_threshold = float(upper_threshold_input) if upper_threshold_input else stats['suggested_upper_threshold']
        
        smooth_input = input("Коэффициент сглаживания (0-1) [0.3]: ").strip()
        smooth_factor = float(smooth_input) if smooth_input else 0.3
        
        min_freq_input = input("Минимальная частота (Гц) [100]: ").strip()
        min_freq = float(min_freq_input) if min_freq_input else 100.0
        
        max_freq_input = input("Максимальная частота (Гц) [4000]: ").strip()
        max_freq = float(max_freq_input) if max_freq_input else 4000.0
        
        speed_input = input("Скорость (%) [100]: ").strip()
        speed_percentage = float(speed_input) if speed_input else 100.0
        
        if lower_threshold >= upper_threshold:
            print("⚠️  Нижний порог должен быть меньше верхнего!")
            lower_threshold = stats['suggested_lower_threshold']
            upper_threshold = stats['suggested_upper_threshold']
        
        params = {
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'smooth_factor': smooth_factor,
            'min_freq': min_freq,
            'max_freq': max_freq,
            'speed_percentage': speed_percentage
        }
        
        return params

    def process_data(self, params):
        print("\n⚙️  Обрабатываю данные...")
        
        original_min = np.min(self.raw_data)
        original_max = np.max(self.raw_data)
        
        if original_max - original_min > 0:
            normalized_all = (self.raw_data - original_min) / (original_max - original_min)
        else:
            normalized_all = np.zeros_like(self.raw_data)
        
        print(f"   • Исходный диапазон: [{original_min:.4f}, {original_max:.4f}] V")
        print(f"   • Нормализован к: [0, 1]")
        
        self.processed_data = normalized_all.copy()
        
        lower_mask = self.processed_data < params['lower_threshold']
        upper_mask = self.processed_data > params['upper_threshold']
        
        self.processed_data[lower_mask] = 0
        self.processed_data[upper_mask] = 1
        
        points_in_range = np.sum((self.processed_data >= params['lower_threshold']) & 
                                (self.processed_data <= params['upper_threshold']))
        
        print(f"   • Нижний порог: {params['lower_threshold']}")
        print(f"   • Верхний порог: {params['upper_threshold']}")
        print(f"   • Точек в диапазоне: {points_in_range:,}")
        
        if params['smooth_factor'] > 0:
            window_size = max(3, int(len(self.processed_data) * params['smooth_factor'] * 0.01))
            if window_size % 2 == 0:
                window_size += 1
            
            if window_size > 1 and window_size < len(self.processed_data):
                window = np.ones(window_size) / window_size
                self.processed_data = np.convolve(self.processed_data, window, mode='same')
                print(f"   • Сглаживание: окно {window_size} точек")
        
        return True

    def generate_audio(self, params):
        """Генерирует аудиосигнал с чистым синусом"""
        print("\n🎵 Генерирую аудио (чистый синус)...")
        
        sample_rate = self.metadata.get('Rate', 1000)
        speed_factor = params['speed_percentage'] / 100.0
        
        target_sample_rate = 44100
        effective_rate = sample_rate * speed_factor
        
        if len(self.processed_data) > 0:
            try:
                num_samples = int(len(self.processed_data) * target_sample_rate / effective_rate)
                if num_samples > 0:
                    resampled_data = signal.resample(self.processed_data, num_samples)
                else:
                    print("❌ Ошибка ресемплирования")
                    return False
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                return False
        else:
            print("❌ Нет данных для обработки")
            return False
        
        frequencies = params['min_freq'] + resampled_data * (params['max_freq'] - params['min_freq'])
        
        t_audio = np.arange(len(frequencies)) / target_sample_rate
        audio_data = np.sin(2 * np.pi * frequencies * t_audio)
        
        # Антиалиасинг-фильтр
        print("   • Применяю антиалиасинг-фильтр...")
        nyquist_freq = target_sample_rate / 2
        cutoff_freq = min(params['max_freq'] * 1.5, nyquist_freq * 0.95)
        
        filter_order = 101
        filter_taps = firwin(filter_order, cutoff_freq, fs=target_sample_rate, window='hamming')
        filtered_audio = lfilter(filter_taps, 1.0, audio_data)
        
        self.audio_data = filtered_audio
        self.audio_data = self.audio_data * 0.5  # уменьшаем громкость
        self.audio_data = np.clip(self.audio_data, -0.99, 0.99)
        
        duration = len(self.audio_data) / target_sample_rate
        print(f"✅ Аудио сгенерировано:")
        print(f"   • Длительность: {duration:.2f} сек")
        print(f"   • Частоты: {params['min_freq']}-{params['max_freq']} Гц")
        print(f"   • Режим: чистый синус (без гармоник)")
        print(f"   • Антиалиасинг: фильтр до {cutoff_freq:.0f} Гц")
        
        return True

    def save_audio(self, filename=None):
        if self.audio_data is None:
            print("❌ Аудио не сгенерировано!")
            return False
        
        if filename is None:
            filename = "output.wav"
        
        try:
            sf.write(filename, self.audio_data, 44100)
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f"💾 Файл сохранен: {filename} ({file_size:.2f} МБ)")
            return True
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False

    def plot_processed_comparison(self, params):
        """Показывает сравнение данных и спектрограмму аудио"""
        if self.raw_data is None or self.processed_data is None or self.audio_data is None:
            print("❌ Данные не обработаны или аудио не сгенерировано!")
            return False
        
        sample_rate = self.metadata.get('Rate', 1000)
        time_axis_raw = np.arange(len(self.raw_data)) / sample_rate
        time_axis_processed = np.arange(len(self.processed_data)) / sample_rate
        
        data_min = np.min(self.raw_data)
        data_max = np.max(self.raw_data)
        if data_max - data_min > 0:
            normalized_raw = (self.raw_data - data_min) / (data_max - data_min)
        else:
            normalized_raw = np.zeros_like(self.raw_data)
        
        in_range_mask = (self.processed_data >= params['lower_threshold']) & (self.processed_data <= params['upper_threshold'])
        
        # 4 графика в одном окне (2x2)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Исходные данные
        ax1.plot(time_axis_raw, self.raw_data, alpha=0.8, linewidth=0.5, color='blue')
        ax1.set_title('1. Исходные данные')
        ax1.set_ylabel('Напряжение (В)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Нормализованные данные с порогами
        ax2.plot(time_axis_raw, normalized_raw, alpha=0.8, linewidth=0.5, color='orange')
        ax2.axhline(y=params['lower_threshold'], color='red', linestyle='--', alpha=0.7, label=f'Нижний порог ({params["lower_threshold"]:.3f})')
        ax2.axhline(y=params['upper_threshold'], color='green', linestyle='--', alpha=0.7, label=f'Верхний порог ({params["upper_threshold"]:.3f})')
        ax2.set_title('2. Нормализованные данные с порогами')
        ax2.set_ylabel('Нормализованное значение')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Данные в пределах порогов
        ax3.plot(time_axis_processed[in_range_mask], self.processed_data[in_range_mask], 
                 alpha=0.8, linewidth=0.5, color='green')
        ax3.set_title('3. Данные в пределах порогов')
        ax3.set_xlabel('Время (секунды)')
        ax3.set_ylabel('Нормализованное значение')
        ax3.grid(True, alpha=0.3)
        
        # 4. Спектрограмма аудио
        if self.audio_data is not None and len(self.audio_data) > 0:
            audio_sample_rate = 44100
            # Вычисляем спектрограмму
            f, t, Sxx = spectrogram(self.audio_data, audio_sample_rate, nperseg=1024, noverlap=512)
            
            # Ограничиваем частотный диапазон для лучшей визуализации
            max_display_freq = params['max_freq'] * 2
            freq_mask = (f >= params['min_freq'] * 0.5) & (f <= max_display_freq)
            f_filtered = f[freq_mask]
            Sxx_filtered = Sxx[freq_mask, :]
            
            # Отображаем спектрограмму в логарифмической шкале
            im = ax4.pcolormesh(t, f_filtered, 10 * np.log10(Sxx_filtered + 1e-10), 
                               shading='gouraud', cmap='viridis')
            ax4.set_title('4. Спектрограмма аудио')
            ax4.set_xlabel('Время (секунды)')
            ax4.set_ylabel('Частота (Гц)')
            
            # Добавляем цветовую шкалу
            plt.colorbar(im, ax=ax4, label='Мощность (дБ)')
            
            # Добавляем информацию о частотном диапазоне
            freq_info = f'Диапазон: {params["min_freq"]}-{params["max_freq"]} Гц'
            
            ax4.axhline(y=params['min_freq'], color='white', linestyle='--', alpha=0.7, linewidth=1)
            ax4.axhline(y=params['max_freq'], color='white', linestyle='--', alpha=0.7, linewidth=1)
            ax4.text(0.02, 0.98, freq_info, 
                     transform=ax4.transAxes, verticalalignment='top', color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return True

    def plot_spectrogram_detail(self, params):
        """Детальная визуализация спектрограммы аудио"""
        if self.audio_data is None:
            print("❌ Аудио не сгенерировано!")
            return False
        
        audio_sample_rate = 44100
        
        # Создаем фигуру с двумя субплoтами
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Временная область аудио
        time_axis = np.arange(len(self.audio_data)) / audio_sample_rate
        ax1.plot(time_axis, self.audio_data, alpha=0.8, linewidth=0.5, color='purple')
        ax1.set_title('Аудиосигнал (временная область)')
        ax1.set_xlabel('Время (секунды)')
        ax1.set_ylabel('Амплитуда')
        ax1.grid(True, alpha=0.3)
        
        # 2. Спектрограмма
        f, t, Sxx = spectrogram(self.audio_data, audio_sample_rate, nperseg=2048, noverlap=1024)
        
        # Ограничиваем частотный диапазон
        max_display_freq = params['max_freq'] * 2
        freq_mask = (f >= params['min_freq'] * 0.5) & (f <= max_display_freq)
        f_filtered = f[freq_mask]
        Sxx_filtered = Sxx[freq_mask, :]
        
        im = ax2.pcolormesh(t, f_filtered, 10 * np.log10(Sxx_filtered + 1e-10), 
                           shading='gouraud', cmap='hot')
        ax2.set_title('Спектрограмма аудио')
        ax2.set_xlabel('Время (секунды)')
        ax2.set_ylabel('Частота (Гц)')
        
        # Цветовая шкала
        plt.colorbar(im, ax=ax2, label='Мощность (дБ)')
        
        # Линии частотного диапазона
        ax2.axhline(y=params['min_freq'], color='cyan', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Мин. частота: {params["min_freq"]} Гц')
        ax2.axhline(y=params['max_freq'], color='magenta', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Макс. частота: {params["max_freq"]} Гц')
        
        ax2.legend()
        
        # Информация о параметрах
        ax2.text(0.02, 0.98, f'Диапазон частот: {params["min_freq"]}-{params["max_freq"]} Гц\n'
                              f'Скорость: {params["speed_percentage"]}%\n'
                              f'Режим: чистый синус', 
                 transform=ax2.transAxes, verticalalignment='top', color='white',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        return True

def main():
    if not check_environment():
        sys.exit(1)
    
    import scipy
    import matplotlib
    import soundfile as sf
    
    print("✅ Зависимости проверены")
    print(f"   • Python: {sys.version.split()[0]}")
    print(f"   • NumPy: {np.__version__}")
    print(f"   • SciPy: {scipy.__version__}")
    print(f"   • Matplotlib: {matplotlib.__version__}")
    print()
    
    sonifier = DataSonifier()
    sonifier.print_banner()
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"📁 Файл: {filename}")
    else:
        filename = input("Введите путь к файлу: ").strip()
    
    filename = filename.strip('"\'')
    
    if not filename:
        print("❌ Путь не указан!")
        return
    
    if not sonifier.load_file(filename):
        return
    
    stats = sonifier.analyze_data()
    if not stats:
        return
    
    print("\n📈 Строю график...")
    if not sonifier.plot_raw_data():
        return
    
    params = sonifier.get_processing_parameters(stats)
    
    if not sonifier.process_data(params):
        return
    
    # Генерация аудио с чистым синусом
    if not sonifier.generate_audio(params):
        return
    
    output_filename = input("Имя файла [output.wav]: ").strip()
    if not output_filename:
        output_filename = "output.wav"
    
    if not output_filename.endswith('.wav'):
        output_filename += '.wav'
    
    sonifier.save_audio(output_filename)
    
    print("\n📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ:")
    print("   1 - Сравнение данных со спектрограммой (4 графика)")
    print("   2 - Детальная спектрограмма аудио")
    print("   3 - Пропустить визуализацию")
    
    viz_choice = input("Выберите вариант [1]: ").strip()
    
    if viz_choice == '2':
        sonifier.plot_spectrogram_detail(params)
    elif viz_choice in ('1', ''):
        sonifier.plot_processed_comparison(params)
    
    print("\n🎉 Преобразование завершено!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Программа прервана")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")