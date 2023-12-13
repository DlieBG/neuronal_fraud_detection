import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import data
import sklearn.ensemble as se

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

train, validate, test = np.split(
    ary=dataframe,
    indices_or_sections=[
        int(.65 * len(dataframe)),
        int(.9 * len(dataframe)),
    ]
)

features = data.numerical_features

X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

y = train['TARGET_BETRUG']

model = se.RandomForestClassifier()
model.fit(X, y)
predictions = model.predict_proba(X_test)

print(predictions)

test['PREDICTION'] = [p[1] for p in predictions]
test.sort_values(by='PREDICTION', ascending=False).to_csv('test.csv')