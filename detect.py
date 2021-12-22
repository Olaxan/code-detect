"""
Usage:
    detect.py [-hreds] [-v NUM] [-m DIR] [-x DIR] [-t DIR] [FILE ...]

Uses deep learning convolutional networks to detect source file contents.

Arguments:
    FILE    A file, the contents of which will be detected

Options:
    -h --help               Show this message
    -r                      Retrain the network
    -e                      Evaluate the trained model against a test set
    -d                      Discard the model efter training (i.e. do not save it)
    -s                      Print a summary of the trained model
    -v NUM --verbose=NUM    Specify the verbosity of the logging (0-3) [default: 1]
    -m DIR --model=DIR      Specify the path to a saved model [default: ./models/model.SavedModel]
    -x DIR --train=DIR      Specify the path to directory labelled training data [default: ./data/train/]
    -t DIR --test=DIR       Specify the path to directory labelled test data [default: ./data/test/]

"""

import os
import sys
import getopt
import shutil
import json
import re

from docopt import docopt

from defs import *

def main(args):
    

    model_path  = args['--model']
    train_path  = args['--train']
    test_path   = args['--test']
    verbose     = args['--verbose']
    retrain     = args['-r']
    evaluate    = args['-e']
    discard     = args['-d']
    summary     = args['-s'] 

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3 - int(verbose))

    has_model = os.path.exists(model_path)

    import tensorflow as tf
    import tensorflow.keras as keras

    from train import setup

    labels = sorted(os.listdir(train_path))
    num_labels = len(labels)

    if retrain:
        _, model = setup(train_path, num_labels)
        if not discard:
            model.save(model_path)
    elif has_model:
        model = keras.models.load_model(model_path)
    else:
        print("No trained model could be located.")
        exit()

    if summary:
        model.summary()

    if evaluate:
        raw_test_ds = keras.utils.text_dataset_from_directory(
                test_path,
                batch_size=VECTOR_BATCH_SIZE)

        perf = model.evaluate(raw_test_ds)
        print("Model performance:", perf)

    texts = []
    files = args['FILE']

    for file in files:
        try:
            with open(file) as f:
                texts.append(f.read())
        except IOError:
            files.remove(file)
            print(f"Error: Failed to open file {file}")

    if len(files) > 0:
        predictions = model.predict(texts)
        predicted_labels = tf.argmax(predictions, axis=1)

        for file, prediction in zip(files, predicted_labels):
            print(f"'{file}', detected language: {labels[prediction].upper()}")




if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)

