from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from  ..utils.helper import *


def ResNet(input_size, filters, kernel_size):
    try:
        inputs = Input(shape=(input_size,))

        x = residual_block(inputs, filters, kernel_size)

        x = residual_block(x, filters, kernel_size)

        outputs = residual_block(x, filters, kernel_size)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        return model
    except:
        print("error")
        return None