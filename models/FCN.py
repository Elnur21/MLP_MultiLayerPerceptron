from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from  ..utils.helper import *


def FCN(input_size, filters, kernel_size):
    try:
        inputs = Input(shape=(input_size,))

        outputs = residual_block(inputs, filters, kernel_size)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        return model
    except:
        print("error")
        return None