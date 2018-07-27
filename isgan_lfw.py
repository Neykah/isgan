"""
ISGAN applied on LFW implemetend using Keras/tensorflow.
Usage: python3 isgan_lfw.py

Authors of the method: Zhang Ru, Shiqi Dong and Liu Jianyi
Implementation: Morgan Lefranc
"""

import numpy as np
from keras import backend as K

from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, Input, MaxPooling2D, AveragePooling2D
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

def rgb2ycc(img_rgb):
    """
    Takes as input a RGB image and convert it to Y Cb Cr space. Shape: channels first.
    """
    output = np.zeros(np.shape(img_rgb))
    output[0, :, :] = 0.299 * img_rgb[0, :, :] + 0.587 * img_rgb[1, :, :] + 0.114 * img_rgb[2, :, :]
    output[1, :, :] = -0.1687 * img_rgb[0, :, :] - 0.3313 * img_rgb[1, :, :] + 0.5 * img_rgb[2, :, :] + 128
    output[2, :, :] = 0.5 * img_rgb[0, :, :] - 0.4187 * img_rgb[1, :, :] + 0.0813 * img_rgb[2, :, :] + 128
    return output


def rgb2gray(img_rgb):
    """
    Transform a RGB image into a grayscale one using weighted method. Shape: channels first.
    """
    output = np.zeros((1, img_rgb.shape[1], img_rgb.shape[2]))
    output[0, :, :] = 0.3 * img_rgb[0, :, :] + 0.59 * img_rgb[1, :, :] + 0.11 * img_rgb[2, :, :]
    return output


class ISGAN(object):
    def __init__(self, img_rows=256, img_cols=256):
        self.img_rows = img_rows
        self.img_cols = img_cols

        #  2 is the Y channel of the cover image concatenated with the grayscale secret image.
        self.gen_input_shape = (2, self.img_rows, self.img_cols)

        self.analyzer_input_shape = (3, self.img_rows, self.img_cols)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the steganalyzer
        self.steganalyzer = self.set_steganalyzer()
        self.steganalyzer.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.set_encoder()
        self.decoder = self.set_decoder()

        imgs_gen = Input(shape=self.gen_input_shape)
        img_analyzer = Input(shape=self.analyzer_input_shape)

        # The generator takes the Y channel of the cover image and the grayscale secret image, fuse them 
        # and reconstruct the secret image from that fusion
        Y_fused = self.encoder(imgs_gen)
        img_fused = K.concatenate((Y_fused, imgs_gen[:, 1:, :, :]), axis=1)
        reconstructed_secret_img = self.decoder(Y_fused)

        # For this adversarial model, we begin by only training the generator
        self.steganalyzer.trainable = False

        # The steganalyzer determines the security of the encoding
        security = self.steganalyzer(img_fused)

        # Adversarial model
        self.adversarial = Model([imgs_gen, img_analyzer], [reconstructed_secret_img, security])
        self.adversarial.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.5, 0.5], optimizer=optimizer)



    def set_encoder(self):
        if self.encoder:
            return self.encoder
        input_shape = (2, self.img_rows, self.img_cols)
        self.encoder = Sequential()

        self.encoder.add(Conv2D(16, 3, input_shape=input_shape, padding='same')) # Layer 1
        self.encoder.add(BatchNormalization(momentum=0.9))
        self.encoder.add(LeakyReLU(alpha=0.2))

        self.encoder.add(InceptionBlock(16, 32))    # Layer 2
        self.encoder.add(InceptionBlock(32, 64))    # Layer 3
        self.encoder.add(InceptionBlock(64, 128))   # Layer 4
        self.encoder.add(InceptionBlock(128, 256))  # Layer 5
        self.encoder.add(InceptionBlock(256, 128))  # Layer 6
        self.encoder.add(InceptionBlock(128, 64))   # Layer 7
        self.encoder.add(InceptionBlock(64, 32))    # Layer 8
        
        self.encoder.add(Conv2D(16, 3, padding='same')) # Layer 9
        self.encoder.add(BatchNormalization(momentum=0.9))
        self.encoder.add(LeakyReLU(alpha=0.2))

        self.encoder.add(Conv2D(1, 1, padding='same', activation='tanh')) # Output

        self.encoder.summary()
        return self.encoder


    def set_decoder(self):
        if self.decoder:
            return self.decoder
        depth = 32
        input_shape = (1, self.img_rows, self.img_cols)
        self.decoder = Sequential()

        # In: 1 x 256 x 256
        # Out: 32 x 256 x 256
        self.decoder.add(Conv2D(depth, 3, input_shape=input_shape, padding='same'))
        self.decoder.add(BatchNormalization(momentum=0.9))
        self.decoder.add(LeakyReLU(alpha=0.2))

        # In: 32 x 256 x 256
        # Out: 64 x 256 x 256
        self.decoder.add(Conv2D(depth*2, 3, padding='same'))
        self.decoder.add(BatchNormalization(momentum=0.9))
        self.decoder.add(LeakyReLU(alpha=0.2))

        # In: 64 x 256 x 256
        # Out: 128 x 256 x 256
        self.decoder.add(Conv2D(depth*4, 3, padding='same'))
        self.decoder.add(BatchNormalization(momentum=0.9))
        self.decoder.add(LeakyReLU(alpha=0.2))

        # In: 128 x 256 x 256
        # Out: 64 x 256 x 256
        self.decoder.add(Conv2D(depth*2, 3, padding='same'))
        self.decoder.add(BatchNormalization(momentum=0.9))
        self.decoder.add(LeakyReLU(alpha=0.2))

        # In: 64 x 256 x 256
        # Out: 32 x 256 x 256
        self.decoder.add(Conv2D(depth, 3, padding='same'))
        self.decoder.add(BatchNormalization(momentum=0.9))
        self.decoder.add(LeakyReLU(alpha=0.2))

        # In: 32 x 256 x 256
        # Out: 1 x 256 x 256
        self.decoder.add(Conv2D(1, 1, padding='same', activation='sigmoid'))
        self.decoder.summary()

        return self.decoder

    def set_steganalyzer(self):
        if self.steganalyzer:
            return self.steganalyzer
        self.steganalyzer = Sequential()
        input_shape = (3, 256, 256)

        self.steganalyzer.add(Conv2D(8, 3, padding='same', input_shape=input_shape))    # Layer 1
        self.steganalyzer.add(BatchNormalization(momentum=0.9))
        self.steganalyzer.add(LeakyReLU(alpha=0.2))
        self.steganalyzer.add(AveragePooling2D(pool_size=5, strides=2, padding='same'))

        self.steganalyzer.add(Conv2D(16, 3, padding='same'))                            # Layer 2
        self.steganalyzer.add(BatchNormalization(momentum=0.9))
        self.steganalyzer.add(LeakyReLU(alpha=0.2))
        self.steganalyzer.add(AveragePooling2D(pool_size=5, strides=2, padding='same'))
                        
        self.steganalyzer.add(Conv2D(32, 1, padding='valid'))                           # Layer 3
        self.steganalyzer.add(BatchNormalization(momentum=0.9))
        self.steganalyzer.add(AveragePooling2D(pool_size=5, strides=2, padding='same'))

        self.steganalyzer.add(Conv2D(64, 1, padding='valid'))                           # Layer 4
        self.steganalyzer.add(BatchNormalization(momentum=0.9))
        self.steganalyzer.add(AveragePooling2D(pool_size=5, strides=2, padding='same'))

        self.steganalyzer.add(Conv2D(128, 3, padding='same'))                           # Layer 5
        self.steganalyzer.add(BatchNormalization(momentum=0.9))
        self.steganalyzer.add(LeakyReLU(alpha=0.2))
        self.steganalyzer.add(AveragePooling2D(pool_size=5, strides=2, padding='same'))

        self.steganalyzer.add(SpatialPyramidPooling([1, 2, 4]))                         # Layer 6

        self.steganalyzer.add(Dense(128))                                               # Layer 7

        self.steganalyzer.add(Dense(2, activation='tanh'))                              # Layer 8

        self.steganalyzer.summary()

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the LFW dataset
        print("Loading the dataset: this step can take a few minutes.")
        lfw_people = fetch_lfw_people(color=True, resize=1.0, slice_=(slice(0, 250), slice(0, 250)))
        images_rgb = lfw_people.images
        images_rgb = np.moveaxis(images_rgb, -1, 1)

        # Convert images from RGB to YCbCr and from RGB to grayscale
        images_ycc = np.zeros(images_rgb.shape)
        images_gray = np.zeros((images_rgb.shape[0], 1, images_rgb.shape[2], images_rgb.shape[3]))
        for k in range(images_rgb.shape[0]):
            images_ycc[k, :, :, :] = rgb2ycc(images_rgb[k, :, :, :])
            images_gray[k, 0, :, :] = rgb2gray(images_rgb[k, :, :, :])

        # Rescale to [-1, 1]
        X_train_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
        X_train_gray = (images_gray.astype(np.float32) - 127.5) / 127.5

        for epoch in range(epochs):
            # Select a random batch of images to use for cover and hide
            cover_idx = np.random.randint(0, X_train_ycc.shape[0], batch_size)
            secret_idx = np.random.randint(0, X_train_gray.shape[0], batch_size)
            cover_imgs = X_train_ycc[cover_idx]
            cover_imgs_Y = cover_imgs[:, 0, :, :]
            secret_imgs = X_train_gray[secret_idx]

            # Concatenate Y channel of cover and secret_img to create input for encoder
            enc_input = np.concatenate((cover_imgs_Y, secret_imgs), axis=1)
            enc_imgs = self.encoder.predict(enc_input)

        

    
if __name__ == "__main__":
    isgan = ISGAN()
    isgan.train(epochs=50, batch_size=32, sample_interval=200)
