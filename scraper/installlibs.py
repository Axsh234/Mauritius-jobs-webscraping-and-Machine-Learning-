import subprocess
import sys

# List of required libraries
required_libs = [
    "requests",
    "beautifulsoup4",
    "mysql-connector-python"
]

# Install each library
for lib in required_libs:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print(f"{lib} installed successfully.")
    except Exception as e:
        print(f"Failed to install {lib}: {e}")

print("All installations attempted.")



import subprocess
import sys

required_libs = [
    "numpy",
    "scikit-learn"
]

for lib in required_libs:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print(f"{lib} installed successfully.")
    except Exception as e:
        print(f"Failed to install {lib}: {e}")

print("All installations attempted.")
