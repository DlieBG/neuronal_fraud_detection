import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

def get_normalization_layer(name: str, dataset: tf.data.Dataset) -> k.layers.Layer:
    normalizer = k.layers.Normalization(axis=None)

    feature_ds = dataset.map(lambda x, _: x[name])

    normalizer.adapt(feature_ds)

    return normalizer

def get_category_encoding_layer(name: str, dataset: tf.data.Dataset, dtype: str, max_tokens: int = None) -> k.layers.Layer:
    if dtype == 'string':
        index = k.layers.StringLookup(max_tokens=max_tokens)
    else:
        index = k.layers.IntegerLookup(max_tokens=max_tokens)

    feature_ds = dataset.map(lambda x, _: x[name])

    index.adapt(feature_ds)

    encoder = k.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    return lambda feature: encoder(index(feature))
