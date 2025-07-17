import keras
from keras import layers

# This is our dataset
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def main():
    # This is the size of our encoded representations
    encoding_dim = 32

    # This is our input image
    input_img = keras.Input(shape=(784,))

    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation = "relu")(input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(784, activation = "sigmoid")(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))

    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    
    # Our network is ready. It is time to set up the other parameters, 
    # using the "adam" optimizer and the "binary_crossentropy" as the loss function
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    # Load MNIST dataset and do preprocessing
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    
    # Print out the sizes of our training and testing set.
    print(x_train.shape)
    print(x_test.shape)

    # Train the model
    autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, 
                    shuffle=True, validation_data=(x_test, x_test))

    # Load up our testing data
    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # See the reconstruction of our digital images
    # Visualizing the digits
    n = 10 # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()