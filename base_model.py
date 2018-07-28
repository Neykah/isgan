import numpy as np

from keras.models import Model
from keras.layers import Conv2D, Input, Reshape, Lambda
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
import keras.layers
from sklearn.datasets import fetch_lfw_people

from utils import InceptionBlock, rgb2gray, rgb2ycc, paper_loss

class BaseModel(object):
    def __init__(self):
        # Generate model
        self.model = self.set_model()

        # Compile model
        # With MSE:
        self.model.compile(optimizer="adam", \
          loss={'enc_output': 'mean_squared_error', 'dec_output': 'mean_squared_error'}, \
          loss_weights={'enc_output': 0.5, 'dec_output': 0.5})

        # Or with custom loss:
        # custom_loss = paper_loss(alpha=0.5, beta=0.3)
        # gamma = 0.85
        # self.model.compile(optimizer="adam", \
        #               loss={'enc_output': custom_loss, 'dec_output': custom_loss}, \
        #               loss_weights={'enc_output': 1, 'dec_output': gamma})

    def set_model(self):
        # Inputs design
        cover_input = Input(shape=(3, 256, 256), name='cover_img')   # cover in YCbCr
        secret_input = Input(shape=(1, 256, 256), name='secret_img') # secret in grayscale

        # Separate Y channel from CbCr channel for cover image
        cover_Y = Lambda(lambda x: x[:, 0, :, :])(cover_input)
        cover_Y = Reshape((1, 256, 256), name="cover_img_Y")(cover_Y)

        cover_cc = Lambda(lambda x: x[:, 1:, :, :])(cover_input)
        cover_cc = Reshape((2, 256, 256), name="cover_img_cc")(cover_cc)

        # Define combined input as combination of Y channel from cover image and secret image
        combined_input = keras.layers.concatenate([cover_Y, secret_input], axis=1)

        # Encoder as defined in Table 1
        L1 = Conv2D(16, 3, padding='same')(combined_input)
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
        enc_output = keras.layers.concatenate([enc_Y_output, cover_cc], axis=1, name="enc_output")

        # Decoder as defined in Table 2
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

        # Build model
        # Inputs are: 
        #   cover image in YCbCr coordinates
        #   secret image in grayscale
        # Outputs are:
        #   stego image in YCbCr coordinates
        #   reconstructed secret image in grayscale

        model = Model(inputs=[cover_input, secret_input], outputs=[enc_output, dec_output])
        model.summary()
        return model
        
    def train(self, epochs, batch_size=4):
        # Load the LFW dataset
        print("Loading the dataset: this step can take a few minutes.")
        # Complete LFW dataset
        # lfw_people = fetch_lfw_people(color=True, resize=1.0, \
        #                               slice_=(slice(0, 250), slice(0, 250)))

        # Smaller dataset used for implementation evaluation
        lfw_people = fetch_lfw_people(color=True, resize=1.0, \
                                      slice_=(slice(0, 250), slice(0, 250)), \
                                      min_faces_per_person=10)

        images_rgb = lfw_people.images
        images_rgb = np.moveaxis(images_rgb, -1, 1)

        # Zero pad them to get 256 x 256 inputs
        images_rgb = np.pad(images_rgb, ((0,0), (0,0), (3,3), (3,3)), 'constant')

        # Convert images from RGB to YCbCr and from RGB to grayscale
        images_ycc = np.zeros(images_rgb.shape)
        images_gray = np.zeros((images_rgb.shape[0], 1, images_rgb.shape[2], images_rgb.shape[3]))
        for k in range(images_rgb.shape[0]):
            images_ycc[k, :, :, :] = rgb2ycc(images_rgb[k, :, :, :])
            images_gray[k, 0, :, :] = rgb2gray(images_rgb[k, :, :, :])

        # Rescale to [-1, 1]
        X_train_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
        X_train_gray = (images_gray.astype(np.float32) - 127.5) / 127.5

        callback = keras.callbacks.ModelCheckpoint(\
                   "base_model/weights.{epoch:02d}-{val_loss:.2f}.hdf5", period=1)
        self.model.fit({'cover_img': X_train_ycc, 'secret_img': X_train_gray}, \
                       {'enc_output': X_train_ycc, 'dec_output': X_train_gray}, \
                       epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[callback])
            

if __name__ == "__main__":
    model = BaseModel()
    model.train(epochs=30)