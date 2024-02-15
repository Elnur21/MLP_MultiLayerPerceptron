from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Dropout



def MultiLayerPerceptron(input_size,num_labels, hidden_units = 500):
    try:
        inputs = Input(shape=(input_size,))

        # Hidden layers
        x = Dense(hidden_units)(inputs)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(hidden_units)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(hidden_units)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Output layer
        outputs = Dense(num_labels, activation='softmax')(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        return model
    except:
        print("error")
        return None