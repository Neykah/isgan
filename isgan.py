import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from keras.models import Model
from keras.layers import Conv2D, Input, AveragePooling2D, Dense, Reshape, Lambda
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
import keras.layers
from sklearn.datasets import fetch_lfw_people
from SpatialPyramidPooling import SpatialPyramidPooling

from utils import InceptionBlock, rgb2gray, rgb2ycc, paper_loss

class ISGAN(object):
    def __init__(self):
        self.images_lfw = None
        # Generate base model
        self.base_model = self.set_base_model()

        # Compile base model
        # With MSE:
        self.base_model.compile(optimizer="adam", \
          loss={'enc_output': 'mean_squared_error', 'dec_output': 'mean_squared_error'}, \
          loss_weights={'enc_output': 0.5, 'dec_output': 0.5})

        # Or with custom loss:
        # custom_loss = paper_loss(alpha=0.5, beta=0.3)
        # gamma = 0.85
        # self.base_model.compile(optimizer="adam", \
        #               loss={'enc_output': custom_loss, 'dec_output': custom_loss}, \
        #               loss_weights={'enc_output': 1, 'dec_output': gamma})

        # Generate discriminator model
        self.discriminator = self.set_discriminator()

        # Compile discriminator
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')

        # Generate adversarial model
        img_cover = Input(shape=(3, 256, 256))
        img_secret = Input(shape=(1, 256, 256))
        imgs_stego, reconstructed_img = self.base_model([img_cover, img_secret])

        # For the adversarial model, we do not train the discriminator
        self.discriminator.trainable = False

        # The discriminator determines the security of the stego image
        security = self.discriminator(imgs_stego)

        # Define a coef for the contribution of discriminator loss to total loss
        delta = 0.5
        # Build and compile the adversarial model
        self.adversarial = Model(inputs=[img_cover, img_secret], \
                                 outputs=[imgs_stego, reconstructed_img, security])
        self.adversarial.summary()
        self.adversarial.compile(optimizer='adam', \
            loss=['mse', 'mse', 'binary_crossentropy'], \
            loss_weights=[0.5, 0.5, 0.5])

    def set_base_model(self):
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

    def set_discriminator(self):
        img_input = Input(shape=(3, 256, 256), name='discrimator_input')
        L1 = Conv2D(8, 3, padding='same')(img_input)
        L1 = BatchNormalization(momentum=0.9)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)
        L1 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L1)

        L2 = Conv2D(16, 3, padding='same')(L1)
        L2 = BatchNormalization(momentum=0.9)(L2)
        L2 = LeakyReLU(alpha=0.2)(L2)
        L2 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L2)

        L3 = Conv2D(32, 1, padding='same')(L2)
        L3 = BatchNormalization(momentum=0.9)(L3)
        L3 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L3)

        L4 = Conv2D(64, 1, padding='same')(L3)
        L4 = BatchNormalization(momentum=0.9)(L4)
        L4 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L4)

        L5 = Conv2D(128, 3, padding='same')(L4)
        L5 = BatchNormalization(momentum=0.9)(L5)
        L5 = LeakyReLU(alpha=0.2)(L5)
        L5 = AveragePooling2D(pool_size=5, strides=2, padding='same')(L5)

        L6 = SpatialPyramidPooling([1, 2, 4])(L5)
        L7 = Dense(128)(L6)
        L8 = Dense(1, activation='tanh', name="D_output")(L7)

        discriminator = Model(inputs=img_input, outputs=L8)
        discriminator.summary()

        return discriminator
        
    def train(self, epochs, batch_size=4):
        # Load the LFW dataset
        print("Loading the dataset: this step can take a few minutes.")
        # Complete LFW dataset
        # lfw_people = fetch_lfw_people(color=True, resize=1.0, \
        #                               slice_=(slice(0, 250), slice(0, 250)))

        # Smaller dataset used for implementation evaluation
        lfw_people = fetch_lfw_people(color=True, resize=1.0, \
                                      slice_=(slice(0, 250), slice(0, 250)), \
                                      min_faces_per_person=3)

        images_rgb = lfw_people.images
        images_rgb = np.moveaxis(images_rgb, -1, 1)

        # Zero pad them to get 256 x 256 inputs
        images_rgb = np.pad(images_rgb, ((0,0), (0,0), (3,3), (3,3)), 'constant')
        self.images_lfw = images_rgb

        # Convert images from RGB to YCbCr and from RGB to grayscale
        images_ycc = np.zeros(images_rgb.shape)
        secret_gray = np.zeros((images_rgb.shape[0], 1, images_rgb.shape[2], images_rgb.shape[3]))
        for k in range(images_rgb.shape[0]):
            images_ycc[k, :, :, :] = rgb2ycc(images_rgb[k, :, :, :])
            secret_gray[k, 0, :, :] = rgb2gray(images_rgb[k, :, :, :])

        # Rescale to [-1, 1]
        X_train_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
        X_train_gray = (secret_gray.astype(np.float32) - 127.5) / 127.5

        # Adversarial ground truths
        original = np.ones((batch_size, 1))
        encrypted = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Select a random batch of cover images
            idx = np.random.randint(0, X_train_ycc.shape[0], batch_size)
            imgs_cover = X_train_ycc[idx]

            # Idem for secret images
            idx = np.random.randint(0, X_train_ycc.shape[0], batch_size)
            imgs_gray = X_train_gray[idx]
            imgs_stego, _ = self.base_model.predict([imgs_cover, imgs_gray])

            # Train the discriminator
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(imgs_cover, original)
            d_loss_encrypted = self.discriminator.train_on_batch(imgs_stego, encrypted)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_encrypted)
            self.discriminator.trainable = False

            # Train the generator
            g_loss = self.adversarial.train_on_batch([imgs_cover, imgs_gray], [imgs_cover, imgs_gray, original])

            # Plot the progress
            print("{} [D loss: {}] [G loss: {}]".format(epoch, d_loss, g_loss[0]))

            self.adversarial.save('adversarial.h5')
            self.discriminator.save('discriminator.h5')
            self.base_model.save('base_model.h5')
    
    def draw_images(self, nb_images=1):
        # Select random images from the dataset
        cover_idx = np.random.randint(0, self.images_lfw.shape[0], nb_images)
        secret_idx = np.random.randint(0, self.images_lfw.shape[0], nb_images)
        imgs_cover = self.images_lfw[cover_idx]
        imgs_secret = self.images_lfw[secret_idx]

        images_ycc = np.zeros(imgs_cover.shape)
        secret_gray = np.zeros((imgs_secret.shape[0], 1, imgs_cover.shape[2], imgs_cover.shape[3]))

        # Convert cover in ycc and secret in gray
        for k in range(nb_images):
            images_ycc[k, :, :, :] = rgb2ycc(imgs_cover[k, :, :, :])
            secret_gray[k, 0, :, :] = rgb2gray(imgs_secret[k, :, :, :])

        # Rescale to [-1, 1]
        X_test_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
        X_test_gray = (secret_gray.astype(np.float32) - 127.5) / 127.5
        
        imgs_stego, imgs_recstr = self.base_model.predict([X_test_ycc, X_test_gray])

        # Unnormalize stego and reconstructed images
        imgs_stego = imgs_stego.astype(np.float32) * 127.5 + 127.5
        imgs_recstr = imgs_recstr.astype(np.float32) * 127.5 + 127.5

        # Flip dimensions of all images to be channel last
        imgs_cover = imgs_cover.transpose((0, 2, 3, 1))
        imgs_stego = imgs_stego.transpose((0, 2, 3, 1))
        secret_gray = np.reshape(secret_gray, (nb_images, 256, 256))
        imgs_recstr = np.reshape(imgs_recstr, (nb_images, 256, 256))


        for k in range(nb_images):
            # plt.imsave('images/cover_{}'.format(k), imgs_cover[k, :, :, :])
            scipy.misc.imsave('images/cover_{}.png'.format(k), imgs_cover[k, :, :, :])
            plt.imsave('images/secret_{}'.format(k), secret_gray[k, :, :], cmap='gray')
            scipy.misc.imsave('images/stego_{}.png'.format(k), imgs_stego[k, :, :, :])
            # plt.imsave('images/stego_{}'.format(k), imgs_stego[k, :, :, :])
            plt.imsave('images/recstr_{}'.format(k), imgs_recstr[k, :, :], cmap='gray')
        
        print("Images drawn.")



if __name__ == "__main__":
    is_model = ISGAN()
    is_model.train(epochs=30)
    is_model.draw_images(5)