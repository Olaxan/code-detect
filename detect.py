"""
Usage:
    detect.py train [-d] [-v NUM] [-m DIR] <PATH>
    detect.py eval [-v NUM] [-m DIR] <PATH>
    detect.py test [-s] [-v NUM] [-m DIR] [<FILE> ...]
    detect.py (-h | --help)
    detect.py --version

Uses deep learning convolutional networks to detect source file contents.

Arguments:
    FILE    A file, the contents of which will be detected by the network.

Options:
    -h --help               Show this message
    --version               Show the program version
    -d                      Discard the model efter training (i.e. do not save it)
    -s                      Print a summary of the trained model
    -v NUM --verbose=NUM    Specify the verbosity of the logging (0-3) [default: 1]
    -m DIR --model=DIR      Specify the path to a saved model [default: ./models/model.SavedModel]
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
    
    train       = args['train']
    evaluate    = args['eval']
    test        = args['test']
    data_path   = args['<PATH>']

    model_path  = args['--model']
    verbose     = args['--verbose']
    discard     = args['-d']
    summary     = args['-s'] 

    files       = args['<FILE>']

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3 - int(verbose))

    import tensorflow as tf
    import tensorflow.keras as keras

    from train import train_model, load_model

    labels = []

    if train:
        model, labels = train_model(data_path, 
                verbose=int(verbose),
                save_path=None if discard else model_path)
        exit()

    model, labels = load_model(model_path)

    if model is None:
        print(f"No model found in {model_path}")
        print(f"Train a model using the 'train' command")
        print(f"Use '--help' for more info")
        exit()

    if evaluate:
        raw_test_ds = keras.utils.text_dataset_from_directory(
                data_path,
                batch_size=VECTOR_BATCH_SIZE)

        perf = model.evaluate(raw_test_ds)
        print("Model performance:", perf)
        exit()


    if summary:
        model.summary()

    if test:
        texts = []

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
    args = docopt(__doc__, version='detect.py version 1.0')
    main(args)

