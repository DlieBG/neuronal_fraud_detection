import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import data, encoder

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

print(class_weights)

print(len(train), 'training examples')
print(len(validate), 'validation examples')
print(len(test), 'test examples')

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

inputs = []
encoded_layers = []

for feature in data.numerical_features + data.datetime_features:
    input = k.Input(
        shape=(1,),
        name=feature,
    )

    normalization_layer = encoder.get_normalization_layer(
        name=feature,
        dataset=train_ds,
    )

    encoded_layer = normalization_layer(input)

    inputs.append(input)
    encoded_layers.append(encoded_layer)
    
for feature in data.categorical_features:
    input = k.Input(
        shape=(1,),
        name=feature,
        dtype='string',
    )

    category_layer = encoder.get_category_encoding_layer(
        name=feature,
        dataset=train_ds,
        dtype='string',
        max_tokens=25,
    )

    encoded_layer = category_layer(input)

    inputs.append(input)
    encoded_layers.append(encoded_layer)
    
input = k.Input(
    shape=(len(data.product_features),),
    name=data.products_feature,
    dtype='string',
)

category_layer = encoder.get_category_encoding_layer(
    name=data.products_feature,
    dataset=train_ds,
    dtype='string',
    max_tokens=100,
    output_mode='multi_hot',
)

encoded_layer = category_layer(input)

inputs.append(input)
encoded_layers.append(encoded_layer)

# create model
x = k.layers.concatenate(encoded_layers)
x = k.layers.Dense(128, activation='relu')(x)
x = k.layers.Dropout(.5)(x)
x = k.layers.Dense(48, activation='relu')(x)
x = k.layers.Dropout(.5)(x)
output = k.layers.Dense(
    1,
    activation='sigmoid',
    bias_initializer=k.initializers.Constant(
        np.log([fraud/non_fraud])
    ),
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

k.utils.plot_model(
    model=model,
    show_shapes=True,
    rankdir='LR',
)

# train model
model.fit(
    x=train_ds,
    epochs=100,
    validation_data=validate_ds,
    class_weight=class_weight_dict,
)

# evaluate model
model.evaluate(
    x=test_ds,
)

# save model
model.save('fraud_classifier.keras')

training_tensors = data.dataframe_to_tensors(dataframe)
dataframe['PREDICTION'] = model.predict(training_tensors)
dataframe.sort_values(by='PREDICTION', ascending=False).to_csv('predict_training.csv')

test_tensors = data.dataframe_to_tensors(test)
test['PREDICTION'] = model.predict(test_tensors)
test.sort_values(by='PREDICTION', ascending=False).to_csv('predict_training_test.csv')
