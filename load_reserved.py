import os
import json

DATA_DIR = "reservedwordsearch/words"
DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_DIR)

print(DATA_PATH)

lang = {}

for filename in os.listdir(DATA_PATH):
    fullpath = os.path.join(DATA_PATH, filename)
    with open(fullpath) as file:
        data = json.load(file)
        name = data["platform"]
        words = [list(word.values())[0] for word in data["words"]]
        lang[name] = words

print(f"Retrieved {len(lang)} languages.")
