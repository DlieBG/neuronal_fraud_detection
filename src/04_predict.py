import tensorflow_decision_forests as tfdf
import sklearn.model_selection as ms
import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import utils.clean, utils.data, utils.encoding_layers, utils.plot

# download dataset
k.utils.get_file(
    fname='classification_dataset.csv',
    origin='https://static.benedikt-schwering.de/code/neuronal_fraud_detection/Klassifizierungsdaten.csv',
    cache_dir='.',
)

dataframe = pd.read_csv(
    filepath_or_buffer='datasets/classification_dataset.csv',
    delimiter=';',
)

dataframe = utils.clean.clean_dataframe(
    dataframe=dataframe,
)

tensors = utils.data.dataframe_to_tensors(
    dataframe=dataframe,
)

model = k.models.load_model('../output/02_neuronal_network/neuronal_network.keras')

dataframe['PREDICTION'] = model.predict(tensors)
dataframe.sort_values(by='PREDICTION', ascending=False).to_csv('predict_classification.csv')
