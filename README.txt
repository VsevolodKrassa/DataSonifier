DataSonifier Installation and Launch Guide

About the Program:

	DataSonifier - a tool for sonification of scientific data

	Convert data from PowerGraph to sound - listen to your experiments!

	Open Source solution from the KVEF art&science research group

Program Installation:

1. Install Python 3.10 or later
Download from the official website: https://www.python.org/downloads/release/python-31011/

Important during installation:

	Check the box "Add Python to PATH" (this is critical!)

	Choose the "Install Now" option

2. Install Dependencies

Windows:

	Open Command Prompt (Win + R → type cmd → Enter)

	Navigate to the program folder:

	cmd
		cd C:\path\to\datasonifier

	Install dependencies:

	cmd
		pip install -r requirements.txt

macOS:

	Open Terminal (Finder → Applications → Utilities → Terminal)

	Navigate to the program folder:

	bash
		cd /path/to/datasonifier
	Install dependencies:

	bash
		pip3 install -r requirements.txt

Linux:

	Open terminal

	Navigate to the program folder:

	bash
		cd /path/to/datasonifier
	Install dependencies:

	bash
		pip3 install -r requirements.txt

3. Verify Installation

Windows:

	cmd
		python check_environment.py

macOS/Linux:

	bash
		python3 check_environment.py

Should display: "Environment is properly configured!"

Launching the Program

Method 1: Specify file at launch (recommended)

Windows:

	cmd
		cd C:\path\to\datasonifier
		py -3.10 datasonifier.py C:\path\to\data\experiment_data.txt

macOS/Linux:

	bash
		cd /path/to/datasonifier
		python3 datasonifier.py /path/to/data/experiment_data.txt

Method 2: Interactive file input

Windows:

	cmd
		cd C:\path\to\datasonifier
		py -3.10 datasonifier.py

	Then enter the file path when prompted by the program

macOS/Linux:

	bash
		cd /path/to/datasonifier
		python3 datasonifier.py

	Then enter the file path when prompted by the program

Input File Format

DataSonifier directly works with annotated data exported from PowerGraph in TXT format with proper metadata headers.

The program expects a text file in the following format:

Block 3
Title:
ADC:	E-440
Module:	E-440
Started:	03/12/2025 11:32:54 PM
Channels:	1
Rate:	1000
Step:	0.001
Duration:	10.0
Size:	10000
Time, s	Channel 1, V
0.000	0.124
0.001	0.135
0.002	0.128
...

Data Conversion (for non-PowerGraph data)

The package includes a converter tool that prepares data from other formats:

1. For L-graph format data
2. For unannotated CSV files

Use the converter to add necessary metadata before processing in DataSonifier:

Windows:
	cmd
		python converter.py your_data.csv

macOS/Linux:
	bash
		python3 converter.py your_data.csv

The converter will guide you through adding ADC information, sampling rate, and other metadata, creating a properly formatted TXT file ready for DataSonifier processing.

Example Path Formats

Windows:

	C:\Users\[username]\Desktop\datasonifier\experiment_data.txt

	D:\Research\data_measurements.txt

macOS:

	/Users/[username]/Documents/datasonifier/experiment_data.txt

	/Desktop/lab_measurements.txt

Linux:

	/home/[username]/datasonifier/experiment_data.txt

	/home/[username]/Downloads/lab_data.txt

Troubleshooting Common Issues

If the python command doesn't work:

	Windows: Use py -3.10 instead of python
	macOS/Linux: Use python3 instead of python

If the pip command doesn't work:

	Windows: Use py -3.10 -m pip
	macOS/Linux: Use python3 -m pip

If the file is not found:

	Make sure the path is correct and the file exists

	Use double quotes if the path contains spaces:

	cmd
		py -3.10 datasonifier.py "C:\My Documents\data file.txt"

If graphs don't display:

	Windows: Tkinter is usually included with Python. If not, reinstall Python with the Tk/Tcl option selected
	macOS/Linux: Install tkinter using your package manager (e.g., `sudo apt-get install python3-tk` on Ubuntu)

What to Do After Launch

1. The program will display a graph of your data - review it and close the window
2. The program will then ask you several questions about the data conversion characteristics

Simple workflow:
  1. The program analyzes and automatically scales the data to a range from 0 to 1, displaying these graphs
  2. You set upper and lower thresholds for data trimming
  3. Configure sound parameters: lowest and highest frequency, smoothing coefficient, and playback speed (can be slowed down or sped up)
  4. Specify a file name - and receive a WAV file with a sinusoidal signal for each recording channel

Use Cases:

1. Sound creation: The resulting WAV file can be processed in any DAW (Digital Audio Workstation like Ableton Live, FL Studio, Logic Pro) or audio programming environments like Max MSP, Pure Data, or SuperCollider to create a full musical composition based on the data
2. Control signal: Set the lowest frequency to 0 Hz and the highest, for example, to 1 Hz, and use the resulting file as a low-frequency control signal (LFO) to modulate parameters of another synthesizer or effect in Max MSP, Pure Data, or modular synthesizers
3. Score generation: Use the pitch information from the audio file to write a score for any acoustic instrument or voice
4. Educational demonstrations: Visualize and sonify scientific phenomena to make them more accessible in educational settings
5. Data exploration: Listen to patterns in data that might not be immediately visible in visual plots

Tips for Best Results:

- Start with small datasets to understand the conversion process
- Experiment with different frequency ranges to find what works best for your data
- Use the scaling and trimming options to focus on the most interesting parts of your data
- Save your parameter settings for reproducible results
