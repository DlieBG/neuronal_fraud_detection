import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import data

# download dataset
k.utils.get_file(
    fname='training_dataset.csv',
    origin='https://static.benedikt-schwering.de/code/neuronal_fraud_detection/Trainingsdaten.csv',
    cache_dir='.',
)

dataframe = pd.read_csv(
    filepath_or_buffer='datasets/training_dataset.csv',
    delimiter=';',
)

dataframe = data.clean_dataframe(
    dataframe=dataframe,
)

train, validate, test = np.split(
    ary=dataframe,
    indices_or_sections=[
        int(.65 * len(dataframe)),
        int(.9 * len(dataframe)),
    ]
)

non_fraud, fraud = np.bincount(dataframe['TARGET_BETRUG'])

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(dataframe['TARGET_BETRUG']), y=dataframe['TARGET_BETRUG'])
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(dataframe['TARGET_BETRUG']), class_weights)}

batch_size = 1024

train_ds = data.dataframe_to_dataset(
    dataframe=train,
    batch_size=batch_size,
)
validate_ds = data.dataframe_to_dataset(
    dataframe=validate,
    batch_size=batch_size,
)
test_ds = data.dataframe_to_dataset(
    dataframe=test,
    batch_size=batch_size,
)

model = tfdf.keras.RandomForestModel()

model.compile(
    metrics=[
        'accuracy',
        k.metrics.TrueNegatives(name='tn'),
        k.metrics.TruePositives(name='tp'),
        k.metrics.FalseNegatives(name='fn'),
        k.metrics.FalsePositives(name='fp'),
    ],
)

model.fit(
    x=train_ds,
    validation_data=validate_ds,
    class_weight=class_weight_dict,
)

model.evaluate(
    x=test_ds,
)

# save model
model.save('fraud_classifier.keras')

features = data.categorical_features + data.numerical_features
# features = [f.name for f in model.make_inspector().features()]


import dtreeviz
viz = dtreeviz.model(
    model=model,
    X_train=dataframe[features],
    y_train=dataframe['TARGET_BETRUG'],
    feature_names=features,
    target_name='TARGET_BETRUG',
    class_names=[0,1]
)

viz.view()

training_tensors = data.dataframe_to_tensors(dataframe)
dataframe['PREDICTION'] = model.predict(training_tensors)
dataframe.sort_values(by='PREDICTION', ascending=False).to_csv('predict_training.csv')

test_tensors = data.dataframe_to_tensors(test)
test['PREDICTION'] = model.predict(test_tensors)
test.sort_values(by='PREDICTION', ascending=False).to_csv('predict_training_test.csv')
