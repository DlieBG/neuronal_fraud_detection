import tensorflow as tf
import keras as k

# load datasets
(x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()

# scale pixels (0 - 255) to (0 - 1)
x_train, x_test = x_train / 255, x_test / 255

# build a model
model = k.models.Sequential([
    k.layers.Flatten(input_shape=(28, 28)),
    k.layers.Dense(132, activation='relu'),
    k.layers.Dropout(.25),
    k.layers.Dense(10),
])

# define loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile model
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'],
)

# train model
model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
)

# evaluate model
print(
    model.evaluate(
        x=x_test,
        y=y_test,
    )
)

# build a probability model
probability_model = k.models.Sequential([
    model,
    k.layers.Softmax(),
])

# save model
probability_model.save('model.tf')
