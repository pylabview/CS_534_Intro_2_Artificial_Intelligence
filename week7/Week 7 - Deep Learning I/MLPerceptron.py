from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # load and partition MNIST dataset
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    # Create partitions of training and testing
    X_train, X_test, y_train, y_test = train_test_split(X/255., y, test_size = 0.20, random_state=1)

    # Build an MLP network
    mlp = MLPClassifier(solver="sgd", max_iter=50, verbose=True, random_state=1, learning_rate_init=0.1
                        ,hidden_layer_sizes=(784, 100, 2))
    # Three hidden layers: 784 neurons, 100 neurons, and 2 neurons
    mlp.fit(X_train, y_train)

    # we predict with our built MLP network
    yhat_train_mlp = mlp.predict(X_train)
    yhat_test_mlp = mlp.predict(X_test)

    print()
    # Training accuracy is usually higher than testing accuracy.
    print(f"Multilayer Perceptron: Accuracy in train is %.2f" % (accuracy_score(y_train, yhat_train_mlp)))
    print(f"Perceptron: Accuracy in test is %.2f" % (accuracy_score(y_test, yhat_test_mlp)))

if __name__ == "__main__":
    main()