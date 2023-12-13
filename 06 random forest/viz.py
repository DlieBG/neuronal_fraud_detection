import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import data

k.utils.get_file(
    fname='training_dataset.csv',
    origin='https://static.benedikt-schwering.de/code/neuronal_fraud_detection/Trainingsdaten.csv',
    cache_dir='.',
)

dataframe = pd.read_csv(
    filepath_or_buffer='datasets/training_dataset.csv',
    delimiter=';',
)

dataframe = data.clean_dataframe(dataframe)

model = k.models.load_model('fraud_classifier.keras')

features = data.categorical_features + data.numerical_features

import dtreeviz
dtreeviz.model(
    model=model,
    X_train=dataframe[features],
    y_train=dataframe['TARGET_BETRUG'],
)