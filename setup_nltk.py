import os

import nltk

# Get current working directory
# current_directory = os.getcwd()
os.makedirs("nltk_data", exist_ok=True)


# # Now download the NLTK data to the current directory
nltk.download('punkt', download_dir="nltk_data")
nltk.download('cmudict', download_dir="nltk_data")