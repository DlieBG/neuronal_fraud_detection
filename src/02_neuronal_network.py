import tensorflow_decision_forests as tfdf
import sklearn.model_selection as ms
import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import utils.clean, utils.data, utils.encoding_layers

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

dataframe = utils.clean.clean_dataframe(
    dataframe=dataframe,
)

# split dataframe
train_df, test_df = ms.train_test_split(dataframe, test_size=.2)
train_df, validate_df = ms.train_test_split(train_df, test_size=.2)

BATCH_SIZE = 1024

train_ds = utils.data.dataframe_to_dataset(
    dataframe=train_df,
    batch_size=BATCH_SIZE,
)
test_ds = utils.data.dataframe_to_dataset(
    dataframe=test_df,
    batch_size=BATCH_SIZE,
)
validate_ds = utils.data.dataframe_to_dataset(
    dataframe=validate_df,
    batch_size=BATCH_SIZE,
)

encoding_layers, inputs = utils.encoding_layers.get_encoding_layers_and_inputs(
    dataset=train_ds,
)

# create model
x = k.layers.concatenate(encoding_layers)
x = k.layers.Dense(128, activation='relu')(x)
x = k.layers.Dropout(.5)(x)
x = k.layers.Dense(32, activation='relu')(x)
x = k.layers.Dropout(.5)(x)
output = k.layers.Dense(
    1,
    activation='sigmoid',
    bias_initializer=utils.data.get_bias_initializer(
        dataframe=dataframe,
    )
)(x)


model = k.Model(inputs, output)

model.compile(
    optimizer=k.optimizers.Adam(
        learning_rate=.001,
    ),
    loss=k.losses.BinaryCrossentropy(
        # label_smoothing=.25,
    ),
    metrics=[
        'accuracy',
        k.metrics.TrueNegatives(name='tn'),
        k.metrics.TruePositives(name='tp'),
        k.metrics.FalseNegatives(name='fn'),
        k.metrics.FalsePositives(name='fp'),
    ],
)

# train model
model.fit(
    x=train_ds,
    epochs=100,
    validation_data=validate_ds,
    class_weight=utils.data.get_class_weights(
        dataframe=dataframe,
    ),
)

# evaluate model
model.evaluate(
    x=test_ds,
)

# save report
with open('neuronal_network_report.txt','w') as file:
    model.summary(print_fn=lambda x: file.write(f'{x} \n'))

# save tree
k.utils.plot_model(
    model=model,
    show_shapes=True,
    rankdir='LR',
    to_file='neuronal_network_model.png',
)

# save model
model.save('neuronal_network.keras')
