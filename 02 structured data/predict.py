import numpy as np
import pandas as pd
import tensorflow as tf
import keras as k

model = k.models.load_model('my_pet_classifier.keras')

sample = {
    'Type': 'Dog',
    'Age': 72,
    'Breed1': 'Mixed Breed',
    'Gender': 'Female',
    'Color1': 'Brown',
    'Color2': 'White',
    'MaturitySize': 'Medium',
    'FurLength': 'Short',
    'Vaccinated': 'Yes',
    'Sterilized': 'Yes',
    'Health': 'Healthy',
    'Fee': 0,
    'PhotoAmt': 3,
}

input_dict = {
    name: tf.convert_to_tensor([value]) for name, value in sample.items()
}

predictions = model.predict(input_dict)
probability = tf.nn.sigmoid(predictions[0])

print(f'The pet has a {probability * 100} percent probability of getting adopted.')
