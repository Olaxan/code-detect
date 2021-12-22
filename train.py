import tensorflow as tf
import tensorflow.keras as keras

from keras import layers
from keras.models import Sequential

from defs import *

def setup(path, num_labels):

    raw_train_ds = keras.utils.text_dataset_from_directory(
            path,
            batch_size=VECTOR_BATCH_SIZE,
            validation_split=0.2,
            subset='training',
            seed=SEED)

    raw_val_ds = keras.utils.text_dataset_from_directory(
            path,
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

    return model, e2e_model
