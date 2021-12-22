import os
import shutil
from pathlib import Path

def make_corpus(input_path, output_path, min_files=4, exclude=[]):

    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(input_path):

        if len(files) == 0:
            continue

        dir_name = os.path.basename(root)
        copy_dir = os.path.join(output_path, dir_name)

        res = shutil.copytree(root, copy_dir, dirs_exist_ok=True)

    for root, dirs, files in os.walk(output_path):

        if len(files) == 0:
            continue

        for file in files:
            full_path = os.path.join(root, file)
            if file in exclude:
                os.remove(full_path)
                files.remove(file)
            else:
                pre, ext = os.path.splitext(full_path)
                os.rename(full_path, pre + '.txt')

        if len(files) < min_files:
            print(f"Too few files ({len(files)}/{min_files}) in {root}, skipping")
            shutil.rmtree(root)
