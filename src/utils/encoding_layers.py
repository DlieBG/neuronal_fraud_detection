import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import utils

def __get_normalization_layer(name: str, dataset: tf.data.Dataset) -> k.layers.Layer:
    normalizer = k.layers.Normalization(axis=None)

    feature_ds = dataset.map(lambda x, _: x[name])

    normalizer.adapt(feature_ds)

    return normalizer

def __get_category_encoding_layer(name: str, dataset: tf.data.Dataset, dtype: str, max_tokens: int = None, output_mode: str = 'int') -> k.layers.Layer:
    if dtype == 'string':
        index = k.layers.StringLookup(
            max_tokens=max_tokens,
            output_mode=output_mode,
        )
    else:
        index = k.layers.IntegerLookup(max_tokens=max_tokens)

    feature_ds = dataset.map(lambda x, _: x[name])

    index.adapt(feature_ds)

    encoder = k.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    return lambda feature: encoder(index(feature))

def get_encoding_layers_and_inputs(dataset: tf.data.Dataset) -> tuple[list[k.Input], list[k.layers.Layer]]:
    encoded_layers = []
    inputs = []

    for feature in utils.clean.numerical_features:
        input = k.Input(
            shape=(1,),
            name=feature,
        )

        normalization_layer = __get_normalization_layer(
            name=feature,
            dataset=dataset,
        )

        encoded_layers.append(normalization_layer(input))
        inputs.append(input)

    for feature in utils.clean.categorical_features:
        input = k.Input(
            shape=(1,),
            name=feature,
            dtype='string',
        )

        category_layer = __get_category_encoding_layer(
            name=feature,
            dataset=dataset,
            dtype='string',
            max_tokens=25,
        )

        encoded_layers.append(category_layer(input))
        inputs.append(input)

    for feature in utils.clean.__datetime_features:
        input = k.Input(
            shape=(1,),
            name=feature['key'],
        )

        category_layer = __get_normalization_layer(
            name=feature['key'],
            dataset=dataset,
        )

        encoded_layers.append(category_layer(input))
        inputs.append(input)
        
    input = k.Input(
        shape=(len(utils.clean.__product_features),),
        name=utils.clean.products_feature,
        dtype='string',
    )

    category_layer = __get_category_encoding_layer(
        name=utils.clean.products_feature,
        dataset=dataset,
        dtype='string',
        max_tokens=100,
        # output_mode='multi_hot',
    )

    encoded_layers.append(category_layer(input))
    inputs.append(input)

    return encoded_layers, inputs
