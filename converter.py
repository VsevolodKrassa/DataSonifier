"""
UNIVERSAL DATA CONVERTER
Version 0.1
Author: KVEF art&science research group
Format: L-graph / CSV → PowerGraph metadata (compatible with datasonifier.py)
"""

import os
import sys
import csv
import re
from datetime import datetime, date, time
from pathlib import Path
import statistics
from typing import List, Tuple, Dict, Optional, Union

class DataConverter:
    """Main class for data conversion"""
    
    def __init__(self):
        self.supported_extensions = ('.txt', '.csv', '.TXT', '.CSV', '.dat', '.DAT')
        self.converted_files = []
        
    def detect_file_format(self, file_path: str) -> str:
        """
        Detects file format based on its content
        
        Returns:
            'lgraph' - L-graph format with "Oscilloscope Data File" header
            'csv' - CSV format with delimiters
            'txt_table' - text file with tabular data
            'unknown' - unknown format
        """
        try:
            # First check the extension
            if file_path.lower().endswith('.csv'):
                return 'csv'
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 2KB for analysis
                content = f.read(2048)
                
                # Check for L-graph format
                if "Oscilloscope Data File" in content:
                    return 'lgraph'
                
                # Analyze file structure
                lines = content.split('\n')[:20]  # Analyze first 20 lines
                
                # Counters for format determination
                csv_like_lines = 0
                table_like_lines = 0
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Check for CSV format (with delimiters)
                    # Count commas, semicolons or tabs
                    if ',' in line or ';' in line or '\t' in line:
                        parts = []
                        if ',' in line:
                            parts = line.split(',')
                        elif ';' in line:
                            parts = line.split(';')
                        elif '\t' in line:
                            parts = line.split('\t')
                        
                        # Check that there are at least 2 columns and they contain numbers
                        if len(parts) >= 2:
                            try:
                                # Try to convert first two columns to numbers
                                float(parts[0].strip())
                                float(parts[1].strip())
                                csv_like_lines += 1
                            except:
                                # If not numbers, check for header
                                if any(keyword in line.lower() for keyword in ['time', 'voltage', 'v,', 'ms,', 'sec']):
                                    csv_like_lines += 2  # Give more weight to lines with headers
                    
                    # Check for table format (space delimiter)
                    elif '  ' in line or line.count(' ') >= 2:
                        parts = re.split(r'\s+', line.strip())
                        if len(parts) >= 2:
                            try:
                                float(parts[0])
                                float(parts[1])
                                table_like_lines += 1
                            except:
                                pass
                
                # Determine format by predominant line type
                if csv_like_lines > table_like_lines:
                    return 'csv'
                elif table_like_lines > 0:
                    return 'txt_table'
                
                # If not determined, check by extension
                if file_path.lower().endswith('.txt'):
                    # For .txt files, check if there is data
                    f.seek(0)
                    data_found = False
                    for line in f:
                        if line.strip() and line.strip()[0].isdigit():
                            data_found = True
                            break
                    if data_found:
                        return 'txt_table'
                
        except Exception as e:
            print(f"  Error detecting format: {e}")
        
        return 'unknown'
    
    def parse_lgraph_file(self, file_path: str) -> Tuple[Dict, List[Tuple[float, float]]]:
        """Parses L-graph format file"""
        print(f"  Reading L-graph file: {file_path}")
        
        # Initialize metadata with placeholders that will be updated
        metadata = {
            'adc': '',
            'started': datetime.now().strftime("%m/%d/%Y %I:%M:%S %p"),
            'rate': 500,
            'channels': 1,
            'duration': 0.0,
            'size': 0,
            'original_rate': 500,
            'original_duration': 0.0,
            'module': '',
            'time_format': 'seconds'
        }
        
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse metadata
            in_data_section = False
            data_start_line = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # If reached data section
                if line == "Data as Time Sequence:":
                    data_start_line = i + 3  # Skip 2 header lines
                    in_data_section = True
                    continue
                
                if in_data_section:
                    # This is data, parse later
                    continue
                
                # Parse metadata
                if line.startswith("Experiment Time"):
                    try:
                        # Format: Experiment Time :   03-12-2025 23:32:54
                        time_str = line.split(":", 1)[1].strip()
                        dt = datetime.strptime(time_str, "%d-%m-%Y %H:%M:%S")
                        metadata['started'] = dt.strftime("%m/%d/%Y %I:%M:%S %p")
                        metadata['experiment_time'] = dt
                    except Exception as e:
                        print(f"  Failed to parse date: {e}")
                
                elif line.startswith("Input Rate In kHz"):
                    try:
                        rate_str = line.split(":", 1)[1].strip()
                        rate_khz = float(rate_str)
                        metadata['rate'] = int(rate_khz * 1000)  # Convert kHz to Hz
                        metadata['original_rate'] = metadata['rate']
                    except Exception as e:
                        print(f"  Failed to parse frequency: {e}")
                
                elif line.startswith("Input Time In Sec"):
                    try:
                        duration_str = line.split(":", 1)[1].strip()
                        metadata['duration'] = float(duration_str)
                        metadata['original_duration'] = metadata['duration']
                    except Exception as e:
                        print(f"  Failed to parse duration: {e}")
                
                elif line.startswith("Number of frames"):
                    try:
                        frames_str = line.split(":", 1)[1].strip()
                        metadata['size'] = int(frames_str)
                    except Exception as e:
                        print(f"  Failed to parse frame count: {e}")
                
                elif line.startswith("Module:"):
                    try:
                        module_str = line.split(":", 1)[1].strip()
                        metadata['module'] = module_str
                        # Store module as is, user will be asked later
                    except Exception as e:
                        print(f"  Failed to parse module: {e}")
                
                elif line.startswith("GPS time="):
                    try:
                        gps_time = line.split("=", 1)[1].strip()
                        if gps_time != "00:00:00 00-00-0000":
                            metadata['gps_time'] = gps_time
                    except:
                        pass
            
            # Parse data
            if data_start_line > 0 and data_start_line < len(lines):
                print(f"Reading data (starting from line {data_start_line + 1})...")
                
                data_lines_processed = 0
                for line in lines[data_start_line:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split line into columns (may have multiple spaces)
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 2:
                        try:
                            time_val = float(parts[0])
                            voltage_val = float(parts[1])
                            data.append((time_val, voltage_val))
                            data_lines_processed += 1
                        except ValueError:
                            continue
                
                print(f"Read data points: {len(data)}")
                
                # Update metadata based on actual data
                if data:
                    metadata['size'] = len(data)
                    
                    # Calculate actual duration
                    if metadata['duration'] <= 0:
                        metadata['duration'] = data[-1][0] - data[0][0]
                    
                    # Check and adjust frequency
                    if len(data) > 1:
                        actual_rate = self._calculate_actual_rate(data)
                        if actual_rate > 0:
                            print(f" Actual frequency from data: {actual_rate:.2f} Hz")
                            
                            # If actual frequency differs significantly from declared
                            if metadata['rate'] > 0 and abs(actual_rate - metadata['rate']) / metadata['rate'] > 0.1:
                                print(f"   Frequency mismatch: declared {metadata['rate']} Hz, actual {actual_rate:.2f} Hz")
                                choice = input(f"  Use actual frequency ({actual_rate:.2f} Hz)? (y/n) [y]: ").strip().lower()
                                if choice != 'n':
                                    metadata['rate'] = actual_rate
            else:
                raise ValueError("Data section not found in file")
            
        except Exception as e:
            raise ValueError(f"Error reading L-graph file: {e}")
        
        return metadata, data
    
    def parse_csv_file(self, file_path: str) -> List[Tuple[float, float]]:
        """Parses CSV file or text file with tabular data"""
        print(f"Reading file: {file_path}")
        
        data = []
        
        # Determine encoding
        encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'windows-1251', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # Read first few lines for analysis
                    sample_lines = []
                    for _ in range(10):
                        line = f.readline()
                        if not line:
                            break
                        sample_lines.append(line)
                    
                    f.seek(0)
                    
                    # Analyze delimiter
                    delimiter = self._detect_delimiter(sample_lines)
                    print(f"  Detected delimiter: '{delimiter}'")
                    
                    # Try to read as CSV
                    reader = csv.reader(f, delimiter=delimiter)
                    
                    # Read header
                    try:
                        headers = next(reader)
                        print(f"Headers: {headers}")
                        
                        # Find time and voltage column indices
                        time_idx, voltage_idx = self._find_columns_indices(headers)
                        print(f"Columns: time[{time_idx}], voltage[{voltage_idx}]")
                        
                        # Read data
                        row_count = 0
                        for row in reader:
                            if len(row) > max(time_idx, voltage_idx):
                                try:
                                    time_val = float(row[time_idx])
                                    voltage_val = float(row[voltage_idx])
                                    data.append((time_val, voltage_val))
                                    row_count += 1
                                except ValueError:
                                    continue
                        
                        if row_count > 0:
                            print(f"Read rows with encoding {encoding}: {row_count}")
                            break
                            
                    except StopIteration:
                        # File without headers
                        f.seek(0)
                        reader = csv.reader(f, delimiter=delimiter)
                        
                        row_count = 0
                        for row in reader:
                            if len(row) >= 2:
                                try:
                                    time_val = float(row[0])
                                    voltage_val = float(row[1])
                                    data.append((time_val, voltage_val))
                                    row_count += 1
                                except ValueError:
                                    continue
                        
                        if row_count > 0:
                            print(f"Read rows (no headers) with encoding {encoding}: {row_count}")
                            break
                            
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with encoding {encoding}: {e}")
                continue
        
        # If standard methods failed, try manual parsing
        if not data:
            print("Standard methods failed, trying manual parsing...")
            data = self._manual_parse_file(file_path)
        
        if not data:
            raise ValueError(f"Failed to extract data from file {file_path}")
        
        print(f"Total data points read: {len(data)}")
        
        return data
    
    def _detect_delimiter(self, sample_lines: List[str]) -> str:
        """Detects delimiter in CSV file"""
        if not sample_lines:
            return ','
        
        # Count frequency of different delimiters
        delimiter_counts = {',': 0, ';': 0, '\t': 0, ' ': 0}
        
        for line in sample_lines:
            delimiter_counts[','] += line.count(',')
            delimiter_counts[';'] += line.count(';')
            delimiter_counts['\t'] += line.count('\t')
            
            # For space, count only if multiple consecutive or between numbers
            if re.search(r'\d\s+\d', line):
                delimiter_counts[' '] += 1
        
        # Select delimiter with maximum frequency
        delimiter = max(delimiter_counts, key=delimiter_counts.get)
        
        # If all counters 0, use comma by default
        if delimiter_counts[delimiter] == 0:
            return ','
        
        return delimiter
    
    def _find_columns_indices(self, headers: List[str]) -> Tuple[int, int]:
        """Finds time and voltage column indices in headers"""
        time_idx = 0
        voltage_idx = 1
        
        time_keywords = ['time', 't,', '(ms)', '(s)', 'sec', 'timestamp']
        voltage_keywords = ['voltage', 'v,', '(v)', 'volt', 'amp', 'current']
        
        for i, header in enumerate(headers):
            header_lower = str(header).lower().strip()
            
            for keyword in time_keywords:
                if keyword in header_lower:
                    time_idx = i
                    break
            
            for keyword in voltage_keywords:
                if keyword in header_lower:
                    voltage_idx = i
                    break
        
        # If voltage_idx matches time_idx, find another column
        if voltage_idx == time_idx and len(headers) > 1:
            voltage_idx = 1 if time_idx != 1 else 2 if len(headers) > 2 else 0
        
        return time_idx, voltage_idx
    
    def _manual_parse_file(self, file_path: str) -> List[Tuple[float, float]]:
        """Manual file parsing when standard methods fail"""
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#') or line.startswith('//'):
                        continue
                    
                    # Try different delimiters
                    for delimiter in [',', ';', '\t', '  ']:
                        if delimiter in line:
                            parts = line.split(delimiter)
                            if len(parts) >= 2:
                                try:
                                    time_val = float(parts[0].strip())
                                    voltage_val = float(parts[1].strip())
                                    data.append((time_val, voltage_val))
                                    break
                                except ValueError:
                                    continue
                    else:
                        # If delimiters not found, try splitting by spaces
                        parts = re.split(r'\s+', line)
                        if len(parts) >= 2:
                            try:
                                time_val = float(parts[0])
                                voltage_val = float(parts[1])
                                data.append((time_val, voltage_val))
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Error during manual parsing: {e}")
        
        return data
    
    def _calculate_actual_rate(self, data: List[Tuple[float, float]]) -> float:
        """Calculates actual sampling frequency from data"""
        if len(data) < 2:
            return 0.0
        
        # Calculate differences between consecutive timestamps
        time_diffs = []
        for i in range(1, min(1000, len(data))):  # Limit for speed
            time_diffs.append(data[i][0] - data[i-1][0])
        
        if not time_diffs:
            return 0.0
        
        # Calculate average time step
        avg_diff = statistics.mean(time_diffs)
        
        # Exclude outliers (steps significantly different from average)
        filtered_diffs = [d for d in time_diffs if 0.5 * avg_diff < d < 2.0 * avg_diff]
        
        if filtered_diffs:
            avg_diff = statistics.mean(filtered_diffs)
        
        if avg_diff <= 0:
            return 0.0
        
        return 1.0 / avg_diff
    
    def get_user_metadata(self, data: List[Tuple[float, float]], 
                         file_name: str, 
                         file_format: str = 'csv',
                         extracted_metadata: Dict = None) -> Tuple[Dict, List[Tuple[float, float]]]:
        """Requests metadata from user"""
        print(f"\n{'='*70}")
        print(f"SETTING PARAMETERS FOR: {file_name}")
        print(f"{'='*70}")
        
        # Initialize default metadata
        if extracted_metadata is None:
            extracted_metadata = {}
        
        # Analyze data
        if data:
            first_time = data[0][0]
            last_time = data[-1][0]
            duration = last_time - first_time
            num_points = len(data)
            
            print(f"DATA ANALYSIS:")
            print(f"First timestamp: {first_time:.6f}")
            print(f"Last timestamp: {last_time:.6f}")
            print(f"Number of points: {num_points:,}")
            print(f"Voltage range: {min(v for _, v in data):.6f} - {max(v for _, v in data):.6f} V")
            
            # Determine time units
            time_unit = self._detect_time_unit(first_time, last_time)
            print(f"Assumed time units: {time_unit}")
            
            # Convert time to seconds if needed
            if time_unit == 'milliseconds':
                print(f"Converting time from milliseconds to seconds...")
                data = [(t/1000.0, v) for t, v in data]
                first_time = data[0][0]
                last_time = data[-1][0]
                duration = last_time - first_time
            
            # Calculate frequency from data
            actual_rate = self._calculate_actual_rate(data)
            if actual_rate > 0:
                print(f"Calculated frequency from data: {actual_rate:.2f} Hz")
                print(f"Average time step: {1.0/actual_rate:.6f} s")
            
            print(f"Data duration: {duration:.3f} s")
        else:
            first_time = 0
            last_time = 0
            duration = 0
            num_points = 0
            actual_rate = 1000
        
        print(f"\n ENTER RECORDING PARAMETERS:")
        
        # 1. Sampling frequency
        default_rate = extracted_metadata.get('rate', actual_rate if actual_rate > 0 else 1000)
        rate = self._get_user_input_float(
            "Sampling frequency (Hz)", 
            default_value=default_rate,
            min_value=0.1,
            max_value=1000000
        )
        
        # 2. ADC module name
        default_adc = extracted_metadata.get('adc', extracted_metadata.get('module', 'E-440'))
        adc = input(f"ADC module name (e.g., E-440, NI-9234, USB-6001) [{default_adc}]: ").strip()
        if not adc:
            adc = default_adc
        
        # 3. Module name (user-defined)
        default_module = extracted_metadata.get('module', adc)
        module = input(f"Module name (custom name for your ADC/module) [{default_module}]: ").strip()
        if not module:
            module = default_module
        
        # 4. Date and time of recording start
        print(f"Recording start date and time:")
        print(f"1. Use current time")
        print(f"2. Enter manually")
        
        time_choice = input("Select option [1]: ").strip()
        
        if time_choice == "2":
            while True:
                try:
                    date_str = input("Enter date (DD-MM-YYYY) [today]: ").strip()
                    if not date_str:
                        today = datetime.now()
                        date_str = today.strftime("%d-%m-%Y")
                    
                    time_str = input("Enter time (HH:MM:SS) [now]: ").strip()
                    if not time_str:
                        now = datetime.now()
                        time_str = now.strftime("%H:%M:%S")
                    
                    dt_str = f"{date_str} {time_str}"
                    dt = datetime.strptime(dt_str, "%d-%m-%Y %H:%M:%S")
                    started = dt.strftime("%m/%d/%Y %I:%M:%S %p")
                    break
                except ValueError:
                    print("Invalid format. Please try again.")
        else:
            started = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
        
        # 5. Number of channels
        default_channels = extracted_metadata.get('channels', 1)
        channels = self._get_user_input_int(
            "Number of channels",
            default_value=default_channels,
            min_value=1,
            max_value=16
        )
        
        # 6. Check and adjust time
        if data and data[0][0] != 0:
            print(f"\n NOTE: Time starts at {data[0][0]:.6f} s")
            print(f" Recommended to shift time to 0 for compatibility with datasonifier.py")
            
            shift_choice = input("Shift time to 0? (y/n) [y]: ").strip().lower()
            if shift_choice != 'n':
                time_offset = data[0][0]
                data = [(t - time_offset, v) for t, v in data]
                print(f"Time shifted. New range: 0.000000 - {data[-1][0]:.6f} s")
        
        # 7. Recalculate duration after possible shift
        if data:
            duration = data[-1][0] - data[0][0]
        
        # Compile metadata
        metadata = {
            'adc': adc,
            'module': module,
            'started': started,
            'rate': rate,
            'channels': channels,
            'duration': duration,
            'size': len(data),
            'original_size': num_points,
            'time_unit_original': time_unit if 'time_unit' in locals() else 'seconds'
        }
        
        # Add any extracted metadata that wasn't overwritten
        for key, value in extracted_metadata.items():
            if key not in metadata:
                metadata[key] = value
        
        return metadata, data
    
    def _detect_time_unit(self, first_time: float, last_time: float) -> str:
        """Determines time measurement units"""
        # If values are large (thousands), likely milliseconds
        if first_time > 1000 or last_time > 10000:
            return 'milliseconds'
        
        # If values are less than 0.1, likely seconds
        if first_time < 0.1 and last_time < 0.1:
            return 'seconds'
        
        # Default to seconds
        return 'seconds'
    
    def _get_user_input_float(self, prompt: str, default_value: float, 
                            min_value: Optional[float] = None, 
                            max_value: Optional[float] = None) -> float:
        """Requests a floating-point number from user"""
        while True:
            try:
                input_str = input(f"  {prompt} [{default_value:.2f}]: ").strip()
                if not input_str:
                    value = default_value
                else:
                    value = float(input_str)
                
                if min_value is not None and value < min_value:
                    print(f" Value must be at least {min_value}")
                    continue
                
                if max_value is not None and value > max_value:
                    print(f" Value must be at most {max_value}")
                    continue
                
                return value
                
            except ValueError:
                print(" Please enter a number")
    
    def _get_user_input_int(self, prompt: str, default_value: int,
                          min_value: Optional[int] = None,
                          max_value: Optional[int] = None) -> int:
        """Requests an integer from user"""
        while True:
            try:
                input_str = input(f"  {prompt} [{default_value}]: ").strip()
                if not input_str:
                    value = default_value
                else:
                    value = int(input_str)
                
                if min_value is not None and value < min_value:
                    print(f" Value must be at least {min_value}")
                    continue
                
                if max_value is not None and value > max_value:
                    print(f" Value must be at most {max_value}")
                    continue
                
                return value
                
            except ValueError:
                print(" Please enter an integer")
    
    def convert_to_powergraph(self, file_path: str, metadata: Dict, 
                            data: List[Tuple[float, float]]) -> str:
        """Converts data to PowerGraph format"""
        # Create output filename
        input_stem = Path(file_path).stem
        output_file = f"converted_{input_stem}.txt"
        
        # Ensure filename is unique
        counter = 1
        while os.path.exists(output_file):
            output_file = f"converted_{input_stem}_{counter}.txt"
            counter += 1
        
        # Prepare metadata for writing
        # Rate must be integer for datasonifier.py
        rate_int = int(round(metadata['rate']))
        
        # Calculate time step
        step = 1.0 / metadata['rate'] if metadata['rate'] > 0 else 0.0
        
        # Prepare duration
        duration = metadata['duration']
        if duration <= 0 and data:
            duration = data[-1][0] - data[0][0]
        
        print(f"\n Saving file: {output_file}")
        print(f" ADC: {metadata['adc']}")
        print(f" Module: {metadata['module']}")
        print(f" Frequency: {rate_int} Hz")
        print(f" Time step: {step:.6f} s")
        print(f" Data points: {metadata['size']:,}")
        print(f" Duration: {duration:.3f} s")
        
        # Write file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("Block 3\n")
                f.write("Title:\t\n")
                f.write(f"ADC:\t{metadata['adc']}\n")
                f.write(f"Module:\t{metadata['module']}\n")
                f.write(f"Started:\t{metadata['started']}\n")
                f.write(f"Channels:\t{metadata['channels']}\n")
                f.write(f"Rate:\t{rate_int}\n")
                f.write(f"Step:\t{step:.6f}\n")
                f.write(f"Duration:\t{duration:.6f}\n")
                f.write(f"Size:\t{metadata['size']}\n")
                f.write("Time, s\tChannel 1, V\n")
                
                # Data
                for time_val, voltage_val in data:
                    f.write(f"{time_val:.6f}\t{voltage_val:.8f}\n")
            
            print(f" File saved successfully!")
            
            # Add to list of converted files
            self.converted_files.append(output_file)
            
            return output_file
            
        except Exception as e:
            raise ValueError(f"Error writing file: {e}")
    
    def process_file(self, file_path: str) -> bool:
        """Processes a single file"""
        print(f"\n{'='*70}")
        print(f" PROCESSING: {file_path}")
        print(f"{'='*70}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f" File not found: {file_path}")
                return False
            
            # Determine file format
            file_format = self.detect_file_format(file_path)
            print(f" Format: {file_format.upper()}")
            
            # Parse file based on format
            if file_format == 'lgraph':
                metadata, data = self.parse_lgraph_file(file_path)
                print(f" L-graph metadata extracted")
                
                # Ask if user wants to override extracted metadata
                print(f"\n Extracted metadata:")
                print(f"  ADC: {metadata.get('adc', 'N/A')}")
                print(f"  Module: {metadata.get('module', 'N/A')}")
                print(f"  Frequency: {metadata.get('rate', 'N/A')} Hz")
                print(f"  Channels: {metadata.get('channels', 'N/A')}")
                
                override = input("\n Override extracted metadata? (y/n) [n]: ").strip().lower()
                if override == 'y':
                    metadata, data = self.get_user_metadata(data, os.path.basename(file_path), file_format, metadata)
                
            elif file_format == 'csv' or file_format == 'txt_table':
                data = self.parse_csv_file(file_path)
                metadata, data = self.get_user_metadata(data, os.path.basename(file_path), file_format)
                
            elif file_format == 'unknown':
                print(f" Failed to determine file format")
                
                # Try to read as CSV/table
                try:
                    data = self.parse_csv_file(file_path)
                    metadata, data = self.get_user_metadata(data, os.path.basename(file_path), 'unknown')
                except Exception as e:
                    print(f" Failed to read file: {e}")
                    return False
            else:
                print(f" Unsupported format: {file_format}")
                return False
            
            # Convert to PowerGraph format
            output_file = self.convert_to_powergraph(file_path, metadata, data)
            
            print(f"\n{'='*70}")
            print(f" SUCCESS: {os.path.basename(file_path)} → {os.path.basename(output_file)}")
            
            return True
            
        except Exception as e:
            print(f"\n{'='*70}")
            print(f" ERROR processing {file_path}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_directory(self, directory_path: str = None):
        """Processes all files in directory"""
        if directory_path is None:
            directory_path = os.getcwd()
        
        print(f"\n SCANNING FOLDER: {directory_path}")
        
        # Find all supported files
        files_to_process = []
        for file in os.listdir(directory_path):
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in self.supported_extensions):
                # Skip already converted files and the script itself
                if not file.startswith('converted_') and file not in ['converter.py', 'converter.exe']:
                    files_to_process.append(os.path.join(directory_path, file))
        
        if not files_to_process:
            print(f" No files found for conversion")
            print(f" Supported formats: {', '.join(self.supported_extensions)}")
            return
        
        print(f" Files found: {len(files_to_process)}")
        for i, file in enumerate(files_to_process, 1):
            file_size = os.path.getsize(file) / 1024
            print(f"    {i:2d}. {os.path.basename(file)} ({file_size:.1f} KB)")
        
        print(f"\n{'='*70}")
        print(f" STARTING CONVERSION...")
        print(f"{'='*70}")
        
        success_count = 0
        
        for file_path in files_to_process:
            file_name = os.path.basename(file_path)
            print(f"\n [{success_count + 1}/{len(files_to_process)}] {file_name}")
            
            if self.process_file(file_path):
                success_count += 1
        
        # Display summary
        print(f"\n{'='*70}")
        print(f" CONVERSION SUMMARY")
        print(f"{'='*70}")
        print(f" Successful: {success_count} of {len(files_to_process)} files")
        
        if success_count > 0:
            print(f"\n CREATED FILES:")
            for i, file in enumerate(self.converted_files, 1):
                file_size = os.path.getsize(file) / 1024
                print(f"    {i:2d}. {os.path.basename(file)} ({file_size:.1f} KB)")
            
            print(f"\n Files are ready for use with datasonifier.py")
            print(f" Use command: python datasonifier.py converted_filename.txt")
        else:
            print(f"\n Failed to convert any files")
        
        return success_count


def main():
    """Main program function"""
    print("\n" + "="*80)
    print(" UNIVERSAL DATA CONVERTER v0.1")
    print(" L-graph / CSV → PowerGraph format (compatible with datasonifier.py)")
    print("="*80)
    print("Authors: KVEF art&science research group")
    print("="*80)
    
    # Create converter instance
    converter = DataConverter()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Process specific file
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            converter.process_file(file_path)
        else:
            print(f" File not found: {file_path}")
            print(f" Check file path")
    else:
        # Process all files in current directory
        converter.process_directory()
    
    # Program completion
    print(f"\n{'='*80}")
    input("Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n Interrupted by user")
    except Exception as e:
        print(f"\n CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")