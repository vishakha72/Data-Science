#importing all libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
#you can use tensorflow.keras instead of keras if you are using updated version of tensorflow

#load mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize data
x_train = x_train/255.0
x_test = x_test/255.0

#Build model
model = models.Sequential([
    layers.Flatten(input_shape = (28,28)),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation ='softmax')
])

#compiling the model
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#Training the model
model.fit(x_train, y_train, epochs = 5)

#Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose =2)

#printing accuracy after running for given epochs
print(f'\nTest accuracy: {test_acc:.4f}')

