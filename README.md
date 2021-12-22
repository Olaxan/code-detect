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

## Usage

Trained models are not included in this repository, but it is quick and easy to create your own.

### Preprocessing
Before the model can be trained, the data must be compiled into a structure Keras can understand. A Keras vectorization layer uses directory names for labels, and all files must have a '.txt' extension.

Use the preprocessor to create a dataset.

`python -m detect preprocess --min-files=N --exclude='README.md' --exclude='testinfo.yml' input_dir/ data_dir/`

The command above will walk through *input_dir/*, moving all bottom-level directories to the *data_dir/*, and changing the extensions of all files within to '.txt'. The original directory is not changed.

Specifying *--min-files=N* means labelled directories with less than N files are skipped.

Specifying *--exclude=FILE* means all occurences of the file will be skipped when preprocessing. This can be useful to ignore readme files and the likes. The option can be supplied many times to ignore multiple files.

### Training

Train the model on the newly processed data by running the detect script with the 'train' parameter.

`python -m detect train data_dir/`
`python -m detect train -m model_path/model.SavedModel data_dir/`

The *-m* or *--model=* options can be used to specify the name of the output model file. The same option must then be used in evaluation and testing mode to find.

Passing *-d* means the model is discarded after training, and not saved. I suppose you can use it to evaluate training performance.

### Evaluating

A dataset of labelled raw strings (non-vectorized) can be used to evaluate the final end-to-end model.

`python -m detect eval test_dir/`

### Testing Files

The finished model can be used to identify files in any of the trained languages.

`python -m detect test testfile.cpp testfile.sh testfile.h`

### Tips

* After training, supported labels are saved in the model directory as 'labels.txt'. These labels can be changed to a more human-readable format without harm (i.e. cplusplus -> C++), as long as the order is maintained.

## Requirements

Due to an apparently long-standing issue with Keras' saving of models with ragged tensors, this program (at the moment of writing) requires *tf-nightly*.

```
docopt==0.6.2
keras-nightly==2.8.0.dev2021122108
Keras-Preprocessing==1.1.2
numpy==1.21.5
pandas==1.3.5
tf-nightly==2.8.0.dev20211221
```
