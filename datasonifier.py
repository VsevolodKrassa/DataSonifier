#!/usr/bin/env python3
"""
DataSonifier - data sonification program
Developed by art&science group KVEF
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
    """Check version compatibility"""
    try:
        actual_tuple = tuple(map(int, actual.split('.')[:3]))
        required_tuple = tuple(map(int, required.split('.')[:3]))
        if actual_tuple != required_tuple:
            print(f" {name} {actual} != {required} (exact match required)")
            return False
        return True
    except Exception as e:
        print(f" Version check error for {name}: {e}")
        return False

def check_environment():
    """Check all dependencies"""
    if sys.version_info < (3, 7):
        print(" DataSonifier requires Python 3.7 or higher!")
        print(f" Current version: {sys.version}")
        return False

    try:
        import numpy as np
        if not check_version(np.__version__, "1.21.6", "NumPy"):
            return False
    except ImportError:
        print(" NumPy is not installed!")
        return False

    try:
        import scipy
        if not check_version(scipy.__version__, "1.7.3", "SciPy"):
            return False
    except ImportError:
        print(" SciPy is not installed!")
        return False

    try:
        import matplotlib
        if not check_version(matplotlib.__version__, "3.5.3", "Matplotlib"):
            return False
    except ImportError:
        print(" Matplotlib is not installed!")
        return False

    try:
        import soundfile as sf
        if not check_version(sf.__version__, "0.12.1", "SoundFile"):
            return False
    except ImportError:
        print(" SoundFile is not installed!")
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
        +=======================================+
        |          DataSonifier v1.0            |
        |       Data to Sound Conversion        |
        |                                       |
        |        Open Source by KVEF            |
        |    art&science research group         |
        +=======================================+
        """
        print(banner)
        print(" Usage: python datasonifier.py [path/to/file.txt]")

    def load_file(self, filename):
        print(f" Loading file: {filename}")
        
        if not os.path.exists(filename):
            print(f" File not found!")
            return False
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.metadata, data_lines = self._parse_metadata(lines)
            self.raw_data = self._parse_data(data_lines)
            
            if len(self.raw_data) == 0:
                print(" No data to process!")
                return False
            
            print(f" Loaded {len(self.raw_data):,} data points")
            if 'Rate' in self.metadata:
                duration = len(self.raw_data) / self.metadata['Rate']
                print(f"   • Duration: {duration:.2f} sec")
            
            return True
        except Exception as e:
            print(f" Error: {e}")
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
            print(" Data not loaded!")
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

        print("\n Data analysis:")
        print(f" Minimum value: {stats['min']:.4f} V")
        print(f" Maximum value: {stats['max']:.4f} V")
        print(f" Mean: {stats['mean']:.4f} V")
        print(f" Standard deviation: {stats['std']:.4f} V")
        print(f" Suggested lower threshold: {stats['suggested_lower_threshold']:.3f}")
        print(f" Suggested upper threshold: {stats['suggested_upper_threshold']:.3f}")
        print(f" Dynamic range: {stats['max'] - stats['min']:.4f} V")
        print(f" Thresholds are set in range [0,1] after normalization")

        return stats

    def plot_raw_data(self):
        if self.raw_data is None:
            print(" Data not loaded!")
            return False
        
        sample_rate = self.metadata.get('Rate', 1000)
        time_axis = np.arange(len(self.raw_data)) / sample_rate
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(time_axis, self.raw_data, alpha=0.8, linewidth=0.5, color='blue')
        ax1.set_title('Raw Data')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True, alpha=0.3)
        
        data_min = np.min(self.raw_data)
        data_max = np.max(self.raw_data)
        
        if data_max - data_min > 0:
            normalized_data = (self.raw_data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(self.raw_data)
        
        ax2.plot(time_axis, normalized_data, alpha=0.8, linewidth=0.5, color='green')
        ax2.set_title('Normalized Data')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Normalized Value [0,1]')
        ax2.grid(True, alpha=0.3)
        
        ax2.text(0.02, 0.98, f'Range: [{data_min:.4f}, {data_max:.4f}] V → [0, 1]', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        return True

    def get_processing_parameters(self, stats):
        print("\n PARAMETER SETUP")
        print("   Thresholds are set in range [0,1]")
        
        lower_threshold_input = input(
            f"Lower threshold [0-1] [recommended {stats['suggested_lower_threshold']:.3f}]: "
        ).strip()
        lower_threshold = float(lower_threshold_input) if lower_threshold_input else stats['suggested_lower_threshold']
        
        upper_threshold_input = input(
            f"Upper threshold [0-1] [recommended {stats['suggested_upper_threshold']:.3f}]: "
        ).strip()
        upper_threshold = float(upper_threshold_input) if upper_threshold_input else stats['suggested_upper_threshold']
        
        smooth_input = input("Smoothing factor (0-1) [0.3]: ").strip()
        smooth_factor = float(smooth_input) if smooth_input else 0.3
        
        min_freq_input = input("Minimum frequency (Hz) [100]: ").strip()
        min_freq = float(min_freq_input) if min_freq_input else 100.0
        
        max_freq_input = input("Maximum frequency (Hz) [4000]: ").strip()
        max_freq = float(max_freq_input) if max_freq_input else 4000.0
        
        speed_input = input("Speed (%) [100]: ").strip()
        speed_percentage = float(speed_input) if speed_input else 100.0
        
        if lower_threshold >= upper_threshold:
            print(" Lower threshold must be less than upper threshold!")
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
        print("\n Processing data...")
        
        original_min = np.min(self.raw_data)
        original_max = np.max(self.raw_data)
        
        if original_max - original_min > 0:
            normalized_all = (self.raw_data - original_min) / (original_max - original_min)
        else:
            normalized_all = np.zeros_like(self.raw_data)
        
        print(f" Original range: [{original_min:.4f}, {original_max:.4f}] V")
        print(f" Normalized to: [0, 1]")
        
        self.processed_data = normalized_all.copy()
        
        lower_mask = self.processed_data < params['lower_threshold']
        upper_mask = self.processed_data > params['upper_threshold']
        
        self.processed_data[lower_mask] = 0
        self.processed_data[upper_mask] = 1
        
        points_in_range = np.sum((self.processed_data >= params['lower_threshold']) & 
                                (self.processed_data <= params['upper_threshold']))
        
        print(f" Lower threshold: {params['lower_threshold']}")
        print(f" Upper threshold: {params['upper_threshold']}")
        print(f" Points in range: {points_in_range:,}")
        
        if params['smooth_factor'] > 0:
            window_size = max(3, int(len(self.processed_data) * params['smooth_factor'] * 0.01))
            if window_size % 2 == 0:
                window_size += 1
            
            if window_size > 1 and window_size < len(self.processed_data):
                window = np.ones(window_size) / window_size
                self.processed_data = np.convolve(self.processed_data, window, mode='same')
                print(f" Smoothing: {window_size} point window")
        
        return True

    def generate_audio(self, params):
        """Generate audio signal with pure sine wave"""
        print("\n Generating audio (pure sine)...")
        
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
                    print(" Resampling error")
                    return False
            except Exception as e:
                print(f" Error: {e}")
                return False
        else:
            print(" No data to process")
            return False
        
        frequencies = params['min_freq'] + resampled_data * (params['max_freq'] - params['min_freq'])
        
        t_audio = np.arange(len(frequencies)) / target_sample_rate
        audio_data = np.sin(2 * np.pi * frequencies * t_audio)
        
        # Anti-aliasing filter
        print("   • Applying anti-aliasing filter...")
        nyquist_freq = target_sample_rate / 2
        cutoff_freq = min(params['max_freq'] * 1.5, nyquist_freq * 0.95)
        
        filter_order = 101
        filter_taps = firwin(filter_order, cutoff_freq, fs=target_sample_rate, window='hamming')
        filtered_audio = lfilter(filter_taps, 1.0, audio_data)
        
        self.audio_data = filtered_audio
        self.audio_data = self.audio_data * 0.5  # reduce volume
        self.audio_data = np.clip(self.audio_data, -0.99, 0.99)
        
        duration = len(self.audio_data) / target_sample_rate
        print(f" Audio generated:")
        print(f" Duration: {duration:.2f} sec")
        print(f" Frequencies: {params['min_freq']}-{params['max_freq']} Hz")
        print(f" Mode: pure sine wave (no harmonics)")
        print(f" Anti-aliasing: filter up to {cutoff_freq:.0f} Hz")
        
        return True

    def save_audio(self, filename=None):
        if self.audio_data is None:
            print(" Audio not generated!")
            return False
        
        if filename is None:
            filename = "output.wav"
        
        try:
            sf.write(filename, self.audio_data, 44100)
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f" File saved: {filename} ({file_size:.2f} MB)")
            return True
        except Exception as e:
            print(f" Error: {e}")
            return False

    def plot_processed_comparison(self, params):
        """Show data comparison and audio spectrogram"""
        if self.raw_data is None or self.processed_data is None or self.audio_data is None:
            print(" Data not processed or audio not generated!")
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
        
        # 4 plots in one window (2x2)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Raw data
        ax1.plot(time_axis_raw, self.raw_data, alpha=0.8, linewidth=0.5, color='blue')
        ax1.set_title('1. Raw Data')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Normalized data with thresholds
        ax2.plot(time_axis_raw, normalized_raw, alpha=0.8, linewidth=0.5, color='orange')
        ax2.axhline(y=params['lower_threshold'], color='red', linestyle='--', alpha=0.7, label=f'Lower threshold ({params["lower_threshold"]:.3f})')
        ax2.axhline(y=params['upper_threshold'], color='green', linestyle='--', alpha=0.7, label=f'Upper threshold ({params["upper_threshold"]:.3f})')
        ax2.set_title('2. Normalized Data with Thresholds')
        ax2.set_ylabel('Normalized Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Data within thresholds
        ax3.plot(time_axis_processed[in_range_mask], self.processed_data[in_range_mask], 
                 alpha=0.8, linewidth=0.5, color='green')
        ax3.set_title('3. Data Within Thresholds')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Normalized Value')
        ax3.grid(True, alpha=0.3)
        
        # 4. Audio spectrogram
        if self.audio_data is not None and len(self.audio_data) > 0:
            audio_sample_rate = 44100
            # Calculate spectrogram
            f, t, Sxx = spectrogram(self.audio_data, audio_sample_rate, nperseg=1024, noverlap=512)
            
            # Limit frequency range for better visualization
            max_display_freq = params['max_freq'] * 2
            freq_mask = (f >= params['min_freq'] * 0.5) & (f <= max_display_freq)
            f_filtered = f[freq_mask]
            Sxx_filtered = Sxx[freq_mask, :]
            
            # Display spectrogram in logarithmic scale
            im = ax4.pcolormesh(t, f_filtered, 10 * np.log10(Sxx_filtered + 1e-10), 
                               shading='gouraud', cmap='viridis')
            ax4.set_title('4. Audio Spectrogram')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Frequency (Hz)')
            
            # Add color bar
            plt.colorbar(im, ax=ax4, label='Power (dB)')
            
            # Add frequency range information
            freq_info = f'Range: {params["min_freq"]}-{params["max_freq"]} Hz'
            
            ax4.axhline(y=params['min_freq'], color='white', linestyle='--', alpha=0.7, linewidth=1)
            ax4.axhline(y=params['max_freq'], color='white', linestyle='--', alpha=0.7, linewidth=1)
            ax4.text(0.02, 0.98, freq_info, 
                     transform=ax4.transAxes, verticalalignment='top', color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return True

    def plot_spectrogram_detail(self, params):
        """Detailed audio spectrogram visualization"""
        if self.audio_data is None:
            print(" Audio not generated!")
            return False
        
        audio_sample_rate = 44100
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Audio time domain
        time_axis = np.arange(len(self.audio_data)) / audio_sample_rate
        ax1.plot(time_axis, self.audio_data, alpha=0.8, linewidth=0.5, color='purple')
        ax1.set_title('Audio Signal (Time Domain)')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 2. Spectrogram
        f, t, Sxx = spectrogram(self.audio_data, audio_sample_rate, nperseg=2048, noverlap=1024)
        
        # Limit frequency range
        max_display_freq = params['max_freq'] * 2
        freq_mask = (f >= params['min_freq'] * 0.5) & (f <= max_display_freq)
        f_filtered = f[freq_mask]
        Sxx_filtered = Sxx[freq_mask, :]
        
        im = ax2.pcolormesh(t, f_filtered, 10 * np.log10(Sxx_filtered + 1e-10), 
                           shading='gouraud', cmap='hot')
        ax2.set_title('Audio Spectrogram')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Frequency (Hz)')
        
        # Color bar
        plt.colorbar(im, ax=ax2, label='Power (dB)')
        
        # Frequency range lines
        ax2.axhline(y=params['min_freq'], color='cyan', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Min frequency: {params["min_freq"]} Hz')
        ax2.axhline(y=params['max_freq'], color='magenta', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Max frequency: {params["max_freq"]} Hz')
        
        ax2.legend()
        
        # Parameter information
        ax2.text(0.02, 0.98, f'Frequency range: {params["min_freq"]}-{params["max_freq"]} Hz\n'
                              f'Speed: {params["speed_percentage"]}%\n'
                              f'Mode: pure sine wave', 
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
    
    print(" Dependencies checked")
    print(f" Python: {sys.version.split()[0]}")
    print(f" NumPy: {np.__version__}")
    print(f" SciPy: {scipy.__version__}")
    print(f" Matplotlib: {matplotlib.__version__}")
    print()
    
    sonifier = DataSonifier()
    sonifier.print_banner()
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f" File: {filename}")
    else:
        filename = input("Enter file path: ").strip()
    
    filename = filename.strip('"\'')
    
    if not filename:
        print(" Path not specified!")
        return
    
    if not sonifier.load_file(filename):
        return
    
    stats = sonifier.analyze_data()
    if not stats:
        return
    
    print("\n Plotting graph...")
    if not sonifier.plot_raw_data():
        return
    
    params = sonifier.get_processing_parameters(stats)
    
    if not sonifier.process_data(params):
        return
    
    # Generate audio with pure sine wave
    if not sonifier.generate_audio(params):
        return
    
    output_filename = input("Output filename [output.wav]: ").strip()
    if not output_filename:
        output_filename = "output.wav"
    
    if not output_filename.endswith('.wav'):
        output_filename += '.wav'
    
    sonifier.save_audio(output_filename)
    
    print("\n VISUALIZATION OPTIONS:")
    print(" 1 - Data comparison with spectrogram (4 plots)")
    print(" 2 - Detailed audio spectrogram")
    print(" 3 - Skip visualization")
    
    viz_choice = input("Choose option [1]: ").strip()
    
    if viz_choice == '2':
        sonifier.plot_spectrogram_detail(params)
    elif viz_choice in ('1', ''):
        sonifier.plot_processed_comparison(params)
    
    print("\n Conversion completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Program interrupted")
    except Exception as e:
        print(f"\n Error: {e}")
        
