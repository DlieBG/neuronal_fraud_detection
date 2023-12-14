import tensorflow_decision_forests as tfdf
import sklearn.model_selection as ms
import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import utils.clean, utils.data

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

BATCH_SIZE = 1024

train_ds = utils.data.dataframe_to_dataset(
    dataframe=train_df,
    batch_size=BATCH_SIZE,
)
test_ds = utils.data.dataframe_to_dataset(
    dataframe=test_df,
    batch_size=BATCH_SIZE,
)

# create model
model = tfdf.keras.GradientBoostedTreesModel()

model.compile(
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
    class_weight=utils.data.get_class_weights(
        dataframe=dataframe,
    ),
)

# evaluate model
loss, acc, tn, tp, fn, fp = model.evaluate(
    x=test_ds,
)

print(f"Evaluate: {loss, acc, tn, tp, fn, fp}")

# save report
with open('gradient_boosted_report.txt','w') as file:
    model.summary(print_fn=lambda x: file.write(f'{x} \n'))

# save trees
for index in [0, 10]:
    with open(f'gradient_boosted_model_{index}.html', 'w+') as file:
        file.write(
            tfdf.model_plotter.plot_model(
                model=model,
                tree_idx=index,
            )
        )

# save model
model.save('gradient_boosted.keras')
