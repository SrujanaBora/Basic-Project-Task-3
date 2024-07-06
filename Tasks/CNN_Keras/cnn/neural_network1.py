from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation

class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        # Initialize the Model
        model = Sequential()

        # Input layer
        model.add(Input(shape=(height, width, depth)))

        # First CONV => RELU => POOL Layer
        model.add(Conv2D(20, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second CONV => RELU => POOL Layer
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Third CONV => RELU => POOL Layer
        model.add(Conv2D(100, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Using Softmax Classifier for Linear Classification
        model.add(Dense(total_classes))
        model.add(Activation("softmax"))

        # If the saved_weights file is already present i.e model is pre-trained, load that weights
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)
                               
        #Print model summary
        model.summary()
            
        return model

