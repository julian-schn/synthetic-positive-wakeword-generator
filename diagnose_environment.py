# Diagnostic cell - Run this in your Jupyter notebook to see what's actually loaded

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print()

# Check if datasets is installed and what version
try:
    import datasets
    print(f"✓ datasets version: {datasets.__version__}")
    print(f"  Location: {datasets.__file__}")
except ImportError as e:
    print(f"✗ datasets not installed: {e}")
print()

# Check pyarrow
try:
    import pyarrow as pa
    print(f"✓ pyarrow version: {pa.__version__}")
    print(f"  Location: {pa.__file__}")
    print(f"  Has PyExtensionType: {hasattr(pa, 'PyExtensionType')}")
    print(f"  Has ExtensionType: {hasattr(pa, 'ExtensionType')}")
except ImportError as e:
    print(f"✗ pyarrow not installed: {e}")
print()

# Check what pip sees
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
print("Installed packages (filtered):")
for line in result.stdout.split('\n'):
    if 'dataset' in line.lower() or 'pyarrow' in line.lower():
        print(f"  {line}")
