import os
import json

def load_reserved(path):

    lang = {}

    for filename in os.listdir(path):
        fullpath = os.path.join(path, filename)
        with open(fullpath) as file:
            data = json.load(file)
            name = data["platform"]
            words = [list(word.values())[0] for word in data["words"]]
            lang[name] = words

    return lang
