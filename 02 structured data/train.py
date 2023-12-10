import numpy as np
import pandas as pd
import tensorflow as tf
import keras as k

# download the dataset
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

k.utils.get_file(
    fname='petfinder_mini.zip',
    origin=dataset_url,
    extract=True,
    cache_dir='.'
)
dataframe = pd.read_csv(csv_file)

# create the target and remove unnecessary fields
dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0, 1)

dataframe = dataframe.drop(
    columns=['AdoptionSpeed', 'Description']
)

# split training, validation and test data
train, validation, test = np.split(
    dataframe.sample(frac=1),
    [
        int(.8 * len(dataframe)),
        int(.9 * len(dataframe)),
    ],
)

print(len(train), 'training examples')
print(len(validation), 'validation examples')
print(len(test), 'test examples')

# dataframe to dataset
def dataframe_to_dataset(dataframe: pd.DataFrame, shuffle: bool = True, batch_size: int = 32) -> tf.data.Dataset:
    df = dataframe.copy()
    labels = df.pop('target')
    
    df = {
        key: np.expand_dims(value, 1) for key, value in dataframe.items()
    }
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)

    return ds

# preprocessing function for numerical features
def get_normalization_layer(name: str, dataset: tf.data.Dataset) -> k.layers.Layer:
    normalizer = k.layers.Normalization(axis=None)

    feature_ds = dataset.map(lambda x, _: x[name])

    normalizer.adapt(feature_ds)

    return normalizer

# preprocessing function for categorical features
def get_category_encoding_layer(name: str, dataset: tf.data.Dataset, dtype: str, max_tokens: int = None) -> k.layers.Layer:
    if dtype == 'string':
        # layer turns string values to int indices
        index = k.layers.StringLookup(max_tokens=max_tokens)
    else:
        # layer turns int values to int indices
        index = k.layers.IntegerLookup(max_tokens=max_tokens)

    feature_ds = dataset.map(lambda x, y: x[name])

    index.adapt(feature_ds)

    encoder = k.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    return lambda feature: encoder(index(feature))

# preprocessing for selected features
numerical_features = ['PhotoAmt', 'Fee']
categorical_string_features = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
categorical_number_features = ['Age']

batch_size = 512

train_ds = dataframe_to_dataset(
    dataframe=train,
    batch_size=batch_size,
)
validation_ds = dataframe_to_dataset(
    dataframe=validation,
    shuffle=False,
    batch_size=batch_size,
)
test_ds = dataframe_to_dataset(
    dataframe=test,
    shuffle=False,
    batch_size=batch_size,
)

all_inputs = []
encoded_features = []

for feature in numerical_features:
    numeric_col = k.Input(
        shape=(1,),
        name=feature,
    )

    normalization_layer = get_normalization_layer(
        name=feature,
        dataset=train_ds,
    )

    encoded_numeric_col = normalization_layer(numeric_col)

    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

for feature in categorical_string_features:
    categorical_string_feature = k.Input(
        shape=(1,),
        name=feature,
        dtype='string',
    )

    encoding_layer = get_category_encoding_layer(
        name=feature,
        dataset=train_ds,
        dtype='string',
        max_tokens=5,
    )

    encoded_categorical_string_feature = encoding_layer(categorical_string_feature)

    all_inputs.append(categorical_string_feature)
    encoded_features.append(encoded_categorical_string_feature)

for feature in categorical_number_features:
    categorical_number_feature = k.Input(
        shape=(1,),
        name=feature,
        dtype='int64',
    )

    encoding_layer = get_category_encoding_layer(
        name=feature,
        dataset=train_ds,
        dtype='int64',
        max_tokens=5,
    )

    encoded_categorical_number_feature = encoding_layer(categorical_number_feature)

    all_inputs.append(categorical_number_feature)
    encoded_features.append(encoded_categorical_number_feature)


# create model
all_features = k.layers.concatenate(encoded_features)

x = k.layers.Dense(48, activation='relu')(all_features)
x = k.layers.Dropout(.75)(x)
output = k.layers.Dense(1)(x)

model = k.Model(all_inputs, output)

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
    validation_data=validation_ds,
)

# evaluate model
loss, accuracy = model.evaluate(
    x=test_ds,
)

print(f'Loss: {loss} Accuracy: {accuracy}')

# save model
model.save('my_pet_classifier.keras')
