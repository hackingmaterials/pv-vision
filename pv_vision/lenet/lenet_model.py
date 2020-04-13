from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as k


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        if k.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        ## Mike recommendation, up the number of filters at this layer.
        model.add(Conv2D(64, (5, 5), padding="same", input_shape=inputShape))
        #model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        ## Reduce filter size to 3? Input is 16x16
        model.add(Conv2D(128, (3, 3), padding="same"))
        #model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #I tried to add one more layers but the result becomes worse
        #model.add(Conv2D(256, (3, 3), padding="same"))
        # model.add(BatchNormalization())
        #model.add(Activation("relu"))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model