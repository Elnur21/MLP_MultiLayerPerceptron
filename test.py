import tensorflow as tf
import pandas as pd

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


from utils.helper import *
from models.MLP import MultiLayerPerceptron


# Load data
df = read_dataset("ArrowHead")


y_train=to_categorical(df[1])


try:
    with tf.device("/device:GPU:0"):
        # Compile the model
        model = MultiLayerPerceptron(input_size=df[0].shape[1], num_labels=y_train.shape[1])
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        # Train the model
        history = model.fit(df[0], y_train, batch_size=32, epochs=10000, validation_split=0.2)
        # Evaluate the model
        loss, accuracy = model.evaluate(df[2], to_categorical(df[3]))
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
except:
    reset_gpu()
    restart_kernel()