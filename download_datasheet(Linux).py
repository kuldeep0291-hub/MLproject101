import os
import subprocess
import sys


# ============================
# CONFIG
# ============================

FILE_ID = "1-Ha8Mx_UPQ9tZ8tTrGsvFozv5DetNDGn"
OUTPUT_NAME = "asvspoof.zip"


# ============================
# PATHS
# ============================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WGET_PATH = os.path.join(PROJECT_DIR, "wget.exe")
OUTPUT_PATH = os.path.join(PROJECT_DIR, OUTPUT_NAME)

DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"


# ============================
# CHECK wget
# ============================

if not os.path.exists(WGET_PATH):
    print("ERROR: wget.exe not found in project folder!")
    print("➡ Download from: https://eternallybored.org/misc/wget/")
    sys.exit(1)


# ============================
# START DOWNLOAD
# ============================

print("===================================")
print("Starting dataset download...")
print("===================================")

command = [
    WGET_PATH,     # Path to wget
    "-c",          # Resume support
    "--show-progress",
    "-O", OUTPUT_PATH,
    DOWNLOAD_URL
]

subprocess.run(command)

print("===================================")
print("Download finished!")
print("Saved as:", OUTPUT_PATH)
print("===================================")
