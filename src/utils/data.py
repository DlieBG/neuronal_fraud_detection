import sklearn.utils.class_weight as cw
import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

import utils

def dataframe_to_tensors(dataframe: pd.DataFrame):
    return {
        feature: tf.convert_to_tensor(dataframe[feature].to_list()) for feature in utils.clean.numerical_features + utils.clean.categorical_features + utils.clean.categorical_list_features
    }
    
def dataframe_to_dataset(dataframe: pd.DataFrame, batch_size: int = 32) -> tf.data.Dataset:
    targets = dataframe.copy().pop('TARGET_BETRUG')
    
    return tf.data.Dataset.from_tensor_slices(
        tensors=(
            dataframe_to_tensors(dataframe),
            targets,
        ),
    ).batch(
        batch_size=batch_size,
    )

def get_class_weights(dataframe: pd.DataFrame) -> dict:
    class_weights = cw.compute_class_weight('balanced', classes=np.unique(dataframe['TARGET_BETRUG']), y=dataframe['TARGET_BETRUG'])
    return { cls: weight for cls, weight in zip(np.unique(dataframe['TARGET_BETRUG']), class_weights) }

def get_bias_initializer(dataframe: pd.DataFrame) -> k.initializers.Initializer:
    non_fraud, fraud = np.bincount(dataframe['TARGET_BETRUG'])

    return k.initializers.Constant(
        np.log([fraud/non_fraud])
    )
