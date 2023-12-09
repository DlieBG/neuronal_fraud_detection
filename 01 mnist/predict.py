import keras as k
import os
from PIL import Image
import numpy as np

model = k.models.load_model('model.tf')

for file in os.listdir('images'):
    img = Image.open(f'images/{file}')
    img.convert('1')
    arr = np.array(img)
    arr = np.array(arr[:,:,0])
    arr = (255 - arr) / 255

    prediction = list(model.predict([arr.tolist()])[0])

    prediction_number = prediction.index(max(prediction))
    print(f'Prediction: {prediction_number} Real: {file}')
