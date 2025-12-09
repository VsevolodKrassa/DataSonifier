#!/usr/bin/env python3
"""
DataSonifier Environment Check Script
"""

import sys

REQUIRED_VERSIONS = {
    'numpy': '1.21.6',
    'scipy': '1.7.3', 
    'matplotlib': '3.5.3',
    'soundfile': '0.12.1'
}

def check_version(actual, required, name):
    """Checks version compatibility"""
    try:
        actual_tuple = tuple(map(int, actual.split('.')[:3]))
        required_tuple = tuple(map(int, required.split('.')[:3]))
        if actual_tuple == required_tuple:
            return True, f" {name}: {actual} - OK"
        else:
            return False, f" {name}: {actual} != {required} (required)"
    except Exception as e:
        return False, f" {name}: version check error - {e}"

def check_environment():
    print(" Checking DataSonifier environment...")
    print()
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python: {python_version}")
    if sys.version_info >= (3, 7):
        print(" Python 3.7+ - OK")
    else:
        print(" Python 3.7+ required")
        return False
    
    print()
    
    # Check libraries
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
            print(f" {package}: NOT INSTALLED")
            all_ok = False
    
    print()
    if all_ok:
        print(" Environment is properly configured!")
        print("Run: python datasonifier.py path/to/file.txt")
    else:
        print(" Environment does not meet requirements")
        print(" Install exact versions: pip install -r requirements.txt")
    
    return all_ok

if __name__ == "__main__":
    check_environment()
    
