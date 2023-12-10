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
        int(.7 * len(dataframe)),
        int(.9 * len(dataframe)),
    ]
)

print(len(train), 'training examples')
print(len(validate), 'validation examples')
print(len(test), 'test examples')

batch_size = 128

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

for feature in data.numerical_features:
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
        max_tokens=100,
    )

    encoded_layer = category_layer(input)

    inputs.append(input)
    encoded_layers.append(encoded_layer)
    
for feature in data.__datetime_features:
    match feature['type']:
        case data.DatetimeType.DATE:
            input = k.Input(
                shape=(3,),
                name=feature['key'],
            )
        case data.DatetimeType.TIME:
            input = k.Input(
                shape=(2,),
                name=feature['key'],
            )

    category_layer = encoder.get_category_encoding_layer(
        name=feature['key'],
        dataset=train_ds,
        dtype='int',
        max_tokens=100,
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
    max_tokens=1000,
)

encoded_layer = category_layer(input)

inputs.append(input)
encoded_layers.append(encoded_layer)

# create model
all_features = k.layers.concatenate(encoded_layers)

x = k.layers.Dense(128, activation='relu')(all_features)
x = k.layers.Dropout(.5)(x)
x = k.layers.Dense(48, activation='relu')(x)
x = k.layers.Dropout(.5)(x)
output = k.layers.Dense(1)(x)

model = k.Model(inputs, output)

model.compile(
    optimizer='adam',
    loss=k.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

k.utils.plot_model(
    model=model,
    show_shapes=True,
    rankdir='LR',
)

# train model
model.fit(
    x=train_ds,
    epochs=10,
    validation_data=validate_ds,
    verbose=2,
)

# evaluate model
loss, accuracy = model.evaluate(
    x=test_ds,
)

print(f'Loss: {loss} Accuracy: {accuracy}')

# save model
model.save('fraud_classifier.keras')

