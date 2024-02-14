import tensorflow as tf
import pandas as pd

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


from utils.helper import *
from utils.constants import *
from models.MLP import MultiLayerPerceptron



results=[]

try:
    with tf.device("/device:GPU:0"):
        for dataset in  UNIVARIATE_DATASET_NAMES_2018:
            # Load data
            df = read_dataset(dataset)

            # apply one-hot encoder
            y_train=to_categorical(df[1])
            y_test=to_categorical(df[3])

            # Compile the model
            model = MultiLayerPerceptron(input_size=df[0].shape[1], num_labels=y_train.shape[1])
            model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

            # Train the model
            history = model.fit(df[0], y_train, batch_size=16, epochs=1000, validation_split=0.2)

            # Evaluate the model
            loss, accuracy = model.evaluate(df[2], y_test)

            results.append(dataset, loss, accuracy)
            
    pd.DataFrame(results, columns=["Dataset","Test loss", "Test accuracy"]).to_csv("result", index=False)
    reset_gpu()

except:
    # reset_gpu()
    # restart_kernel()
    print("olmur")