import os
import tensorflow as tf
import tensorflow.keras as keras

from keras import layers
from keras.models import Sequential

from defs import *

def save_labels(labels, save_path):
    try:
        with open(save_path, 'w') as f:
            f.write('\n'.join(labels))
    except IOError:
        print(f"Error: Could not open file '{save_path}' for writing.")

def load_labels(load_path):

    lines = []

    try:
        with open(load_path, 'r') as f:
            lines = f.readlines()
    except IOError:
        print(f"Error: Could not open file '{load_path}' for reading.")

    return lines

def train_model(train_path, save_path):

    labels = sorted(os.listdir(train_path))
    num_labels = len(labels)

    raw_train_ds = keras.utils.text_dataset_from_directory(
            train_path,
            batch_size=VECTOR_BATCH_SIZE,
            validation_split=0.2,
            subset='training',
            seed=SEED)

    raw_val_ds = keras.utils.text_dataset_from_directory(
            train_path,
            batch_size=VECTOR_BATCH_SIZE,
            validation_split=0.2,
            subset='validation',
            seed=SEED)

    vectorize_layer = layers.TextVectorization(
            standardize=None,
            max_tokens=MAX_FEATURES, 
            output_mode='int', 
            output_sequence_length=SEQUENCE_LENGTH)

    text_ds = raw_train_ds.map(lambda x, y: x)

    vectorize_layer.adapt(text_ds)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)

    train_ds = train_ds.cache().prefetch(buffer_size=BUFFER_SIZE)
    val_ds = val_ds.cache().prefetch(buffer_size=BUFFER_SIZE)

    inputs = keras.Input(shape=(None,), dtype='int64')

    x = layers.Embedding(MAX_FEATURES, EMBEDDINGS_DIM)(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
    x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    predictions = layers.Dense(num_labels, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, predictions)

    model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy'])

    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=NUM_EPOCHS,
            callbacks=[es])

    inputs = keras.Input(shape=(1,), dtype='string')
    indices = vectorize_layer(inputs)
    output = model(indices)

    e2e_model = keras.Model(inputs, output)
    e2e_model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy'])

    if save_path is not None:
        e2e_model.save(save_path)
        labels_path = os.path.join(save_path, LABELS_NAME)
        save_labels(labels, labels_path)

    return e2e_model, labels

def load_model(path):
    
    model = keras.models.load_model(path)
    labels_path = os.path.join(path, LABELS_NAME)
    labels = load_labels(labels_path)

    return model, labels
