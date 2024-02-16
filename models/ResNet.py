from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense

from  utils.helper import *


def ResNet(input_size, nb_classes):
    try:
        inputs = Input(shape=input_size)

        x_skip=inputs
        x = residual_block(inputs, [64,64,64])
        if x_skip.shape[-1] != 64:
            x_skip = Conv1D(64, kernel_size=1, padding='same')(x_skip)
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)

        x_skip=x
        x = residual_block(x, [128,128,128])
        if x_skip.shape[-1] != 128:
            x_skip = Conv1D(128, kernel_size=1, padding='same')(x_skip)
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)

        x_skip=x
        x = residual_block(x, [128,128,128])
        if x_skip.shape[-1] != 128:
            x_skip = Conv1D(128, kernel_size=1, padding='same')(x_skip)
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)

        x = GlobalAveragePooling1D()(x)

        outputs = Dense(nb_classes, activation='softmax')(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        return model
    except Exception as e:
        print(e)
        return None