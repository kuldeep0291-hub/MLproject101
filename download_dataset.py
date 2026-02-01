#This python file allows us to download the test cases from the internet.
import os
import gdown
import zipfile

# File ID
FILE_ID = "1-Ha8Mx_UPQ9tZ8tTrGsvFozv5DetNDGn"

# Output zip name
OUTPUT_FILE = "asvspoof.zip"

# Drive URL
URL = f"https://drive.google.com/file/d/{FILE_ID}/view"

# Project folder
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) #Tells the o/s where the folder of the python file is.
DOWNLOAD_PATH = os.path.join(PROJECT_DIR, OUTPUT_FILE) #Tells the o/s where to store the file and by what name. Actually what it does is joins the path PROJECT_DIR to OUTPUT_FILE by p_d/O_F
# ============================
# Download 
# ============================
print("Downloading dataset...")

gdown.download(URL, DOWNLOAD_PATH, quiet=False, fuzzy=True) #quiet=false means to show the progress bar and fuzzy is to try(by force is gdrive is preventing the download. and it actually downloads the file.

print("Download completed!")

# ============================
# Extract
# ============================
if zipfile.is_zipfile(DOWNLOAD_PATH): #it checks if the file is zip

    print("Extracting files...")

    with zipfile.ZipFile(DOWNLOAD_PATH, "r") as zip_ref:
        zip_ref.extractall(PROJECT_DIR) #stores the file as zip_ref by extracting it.

    print("Extraction finished!")

else:
    print("Not a zip file!") #if not a zip file it runs.

print("All done!") #Bravo our work is done.
