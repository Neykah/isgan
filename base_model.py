import numpy as np
from keras import backend as K

from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, Input, MaxPooling2D, AveragePooling2D, Reshape
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import keras.layers
from sklearn.datasets import fetch_lfw_people

from SpatialPyramidPooling import SpatialPyramidPooling

def InceptionBlock(filters_in, filters_out):
    input_layer = Input(shape=(filters_in, 256, 256))
    tower_filters = int(filters_out / 4)

    tower_1 = Conv2D(tower_filters, 1, padding='same', activation='relu')(input_layer)

    tower_2 = Conv2D(tower_filters, 1, padding='same', activation='relu')(input_layer)
    tower_2 = Conv2D(tower_filters, 3, padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(tower_filters, 1, padding='same', activation='relu')(input_layer)
    tower_3 = Conv2D(tower_filters, 5, padding='same', activation='relu')(tower_3)

    tower_4 = MaxPooling2D(tower_filters, padding='same', strides=(1, 1))(input_layer)
    tower_4 = Conv2D(tower_filters, 1, padding='same', activation='relu')(tower_4)

    concat = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=1)

    res_link = Conv2D(filters_out, 1, padding='same', activation='relu')(input_layer)

    output = keras.layers.add([concat, res_link])
    output = Activation('relu')(output)

    model_output = Model([input_layer], output)
    return model_output


class BaseModel(object):
    def __init__(self):
        # Inputs design
        self.cover_input = Input(shape=(3, 256, 256), name='cover_img')   # cover in YCbCr
        self.secret_input = Input(shape=(1, 256, 256), name='secret_img') # secret in grayscale

        # self.cover_Y = Reshape((1, self.cover_input.shape[2], self.cover_input.shape[3]))(self.cover_input[:,0,:,:])
        self.cover_Y = Input(shape=(1, 256, 256), name='cover_img_Y')

        self.combined_input = keras.layers.concatenate([self.cover_Y, self.secret_input], axis=1)

        # Encoder
        L1 = Conv2D(16, 3, padding='same')(self.combined_input)
        L1 = BatchNormalization(momentum=0.9)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)

        L2 = InceptionBlock(16, 32)(L1)
        L3 = InceptionBlock(32, 64)(L2)
        L4 = InceptionBlock(64, 128)(L3)
        L5 = InceptionBlock(128, 256)(L4)
        L6 = InceptionBlock(256, 128)(L5)
        L7 = InceptionBlock(128, 64)(L6)
        L8 = InceptionBlock(64, 32)(L7)

        L9 = Conv2D(16, 3, padding='same')(L8)
        L9 = BatchNormalization(momentum=0.9)(L9)
        L9 = LeakyReLU(alpha=0.2)(L9)

        enc_Y_output = Conv2D(1, 1, padding='same', activation='tanh', name="enc_Y_output")(L9)
        # enc_output = keras.layers.concatenate([enc_Y_output, Reshape((2, self.cover_input.shape[2], self.cover_input.shape[3]))(self.cover_input[:, 1:, :, :])], axis=1)

        print("Enc_Y_output shape: ", enc_Y_output.shape)
        # print("Enc_output shape: ", enc_output.shape)

        # Decoder
        depth = 32
        L1 = Conv2D(depth, 3, padding='same')(enc_Y_output)
        L1 = BatchNormalization(momentum=0.9)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)

        L2 = Conv2D(depth*2, 3, padding='same')(L1)
        L2 = BatchNormalization(momentum=0.9)(L2)
        L2 = LeakyReLU(alpha=0.2)(L2)

        L3 = Conv2D(depth*4, 3, padding='same')(L2)
        L3 = BatchNormalization(momentum=0.9)(L3)
        L3 = LeakyReLU(alpha=0.2)(L3)

        L4 = Conv2D(depth*2, 3, padding='same')(L3)
        L4 = BatchNormalization(momentum=0.9)(L4)
        L4 = LeakyReLU(alpha=0.2)(L4)

        L5 = Conv2D(depth, 3, padding='same')(L4)
        L5 = BatchNormalization(momentum=0.9)(L5)
        L5 = LeakyReLU(alpha=0.2)(L5)

        dec_output = Conv2D(1, 1, padding='same', activation='sigmoid', name="dec_output")(L5)

        print ("dec_output_shape: ", dec_output.shape)

        # Build model
        self.model = Model(inputs=[self.cover_Y, self.secret_input], outputs=[enc_Y_output, dec_output])
        self.model.summary()

        # Compile model
        self.model.compile(optimizer="adam", \
                      loss={'enc_Y_output': 'mean_squared_error', 'dec_output': 'mean_squared_error'}, \
                      loss_weights={'enc_Y_output': 0.5, 'dec_output': 0.5})
        





if __name__ == "__main__":
    model = BaseModel()