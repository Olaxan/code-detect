import os
import shutil
from pathlib import Path

FILE_DIR = os.path.dirname(__file__)
INPUT_DIR = "deep-learning-lang-detection/data/test" #'sample-programs/archive'
INPUT_PATH = os.path.join(FILE_DIR, INPUT_DIR)
OUTPUT_DIR = "test/"
OUTPUT_PATH= os.path.join(FILE_DIR, OUTPUT_DIR)

REMOVE_FILES = ['README.md', 'testinfo.yml']
MIN_FILES = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

for root, dirs, files in os.walk(INPUT_PATH):

    if len(files) == 0:
        continue

    dir_name = os.path.basename(root)
    copy_dir = os.path.join(OUTPUT_PATH, dir_name)

    res = shutil.copytree(root, copy_dir, dirs_exist_ok=True)

for root, dirs, files in os.walk(OUTPUT_PATH):

    if len(files) == 0:
        continue

    for file in files:
        full_path = os.path.join(root, file)
        if file in REMOVE_FILES:
            os.remove(full_path)
            files.remove(file)
        else:
            pre, ext = os.path.splitext(full_path)
            os.rename(full_path, pre + '.txt')

    if len(files) < MIN_FILES:
        print(f"Warning: Too few files ({len(files)}) in {root}")
        shutil.rmtree(root)
