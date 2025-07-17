from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import chainer

def main():
    # Load the MNIST dataset, same as mnist_784, from pre-inn chainer method to plot how images look like.
    train, test = chainer.datasets.get_mnist(ndim=1)
    ROW = 4
    COLUMN = 5
    for i in range(ROW * COLUMN):
        # train[i][0] is i-th image data with size 28x28
        image = train[i][0].reshape(28, 28)  # not necessary to reshape if ndim is set to 2
        plt.subplot(ROW, COLUMN, i + 1)  # subplot with size (width 3, height 5)
        plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
        # train[i][1] is i-th digit label
        plt.title('label = {}'.format(train[i][1]))
        plt.axis('off')  # do not show axis value
    plt.tight_layout()  # automatic padding between subplots
    plt.savefig('mnist_plot.png')
    plt.show()

    # load and partition MNIST dataset
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # you can check how if we change random_state (seed for test/train split)
    # the accuracy of our models also change!

    X_train, X_test, y_train, y_test = train_test_split(X/255., y, test_size = 0.20, random_state=1)

    # first let's use a very simple linear perceptron: we set the hyperparams and train
    # a perceptron does not use activation units which means it's a completely linear model
    per = Perceptron(random_state=1, max_iter=50, tol=0.005)

    per.fit(X_train, y_train)

    # we predict with our built perceptron
    yhat_train_per = per.predict(X_train)
    yhat_test_per = per.predict(X_test)

    # Training accuracy is usually higher than testing accuracy.
    print(f"Perceptron: Accuracy in train is %.2f" % (accuracy_score(y_train, yhat_train_per)))
    print(f"Perceptron: Accuracy in test is %.2f" % (accuracy_score(y_test, yhat_test_per)))

if __name__ == "__main__":
    main()