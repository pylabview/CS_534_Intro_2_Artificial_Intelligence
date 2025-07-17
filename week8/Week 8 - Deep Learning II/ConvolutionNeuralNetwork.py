import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    # Load training and testing data from the built-in dataset CIFAR.
    # The CIFAR-10 dataset consists of 60,000 32 x 32 color images in 10 classes,
    # with 6000 images per class. There are 50,000 training images and 10,000 test images

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Each of the images contains with 8-bit color information.
    # Let us normalize that information to a value between 0 and 1 by dividing it by 255.
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Also, we will initialize the class labels.
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Let us take a look at our training data. The following loop will display the first 25 images with their corresponding labels.
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])

        # The CIFAR labels happen to be arrays
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

    # Build the CNN-based model with several multidimensional convolution layers
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation = "relu"))
    model.add(tf.keras.layers.Dense(10))

    # Define how we are going to compile the model, how we
    # will treat the error and how we will measure the success
    model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ["accuracy"])

    # Retain the history of training epochs
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # At this point, our training and testing are done, and we are ready to plot and report both of these phases:
    plt.plot(history.history["accuracy"], label = "accuracy")
    plt.plot(history.history["val_accuracy"], label = "val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc = "lower right")
    plt.show()

    # For each of the 10 epochs, along with corresponding accuracy and loss values, we get a plot for training and testing accuracies
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print()
    print("Test Loss: ", test_loss)
    print("Test Accuracy: ", test_acc)

if __name__ == "__main__":
    main()