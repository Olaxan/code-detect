# Detect.py
## CNN Language Detection

Author: Fredrik Lind  
Mail: 	fredrik.lind.96@gmail.com

Written for UMU 5TF078 Deep Learning.

This program can identify languages using Keras text embeddings and convolutional neural networks. It was written with programming language identification in mind, but should work with any natural language.

A basic data preprocessor is included to compile training/testing datasets.

```
Usage:
    detect.py preprocess [--min-files=<NUM>] [--exclude=<FILE ...>] <INPUT> <OUTPUT>
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
    --min-files=<NUM>       Specify the minimum number of examples for the preprocessor to include in the corpus [default: 1]
    --exclude=<FILE>        Specify files to exclude from the corpus
```

## 
