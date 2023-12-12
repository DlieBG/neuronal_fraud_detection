import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import data

# download dataset
k.utils.get_file(
    fname='classification_dataset.csv',
    origin='https://static.benedikt-schwering.de/code/neuronal_fraud_detection/Klassifizierungsdaten.csv',
    cache_dir='.',
)

dataframe_classification = pd.read_csv(
    filepath_or_buffer='datasets/classification_dataset.csv',
    delimiter=';',
)

dataframe_classification = data.clean_dataframe(
    dataframe=dataframe_classification,
)

tensors_classification = data.dataframe_to_tensors(
    dataframe=dataframe_classification,
)

model = k.models.load_model('fraud_classifier.keras')

dataframe_classification['PREDICTION'] = model.predict(tensors_classification)
dataframe_classification.sort_values(by='PREDICTION', ascending=False).to_csv('predict_classification.csv')
