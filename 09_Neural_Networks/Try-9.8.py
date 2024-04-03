import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

cifar10 = tf.keras.datasets.cifar10

(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()

training_images = training_images.reshape(50000, 1024, 3)

training_images = training_images.reshape(50000, 1024, 3)
training_images = training_images/255.0
test_images = test_images.reshape(10000, 1024, 3)
test_images = test_images/255.0

model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(1024, 3), return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=["acc"])
model.fit(training_images, training_labels, batch_size = 50, epochs=20)
