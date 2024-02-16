from utils.helper import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Activation, Dense



def FCN(input_size,nb_classes, filters):
    try:
        inputs = Input(shape=input_size)
        x = residual_block(inputs, filters)

        x = GlobalAveragePooling1D()(x)

        outputs = Dense(nb_classes, activation='softmax')(x)


        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        return model
    except Exception as e:
        print(e)
        return None