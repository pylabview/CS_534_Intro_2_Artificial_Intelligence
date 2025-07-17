import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def main():
    # Load our dataset
    raw_data = "seeds_dataset.txt"
    dataset = np.loadtxt(raw_data)
    print("Dataset:")
    print(dataset)
    print("\nShape of Dataset: ", dataset.shape) # We have 210 data points and the 8 features listed above.

    # Create partitions of training and testing
    # The first seven columns are features and the 8th column is the target
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0:7], dataset[:, 7],
                                                        test_size=0.25, random_state=42)

    # 5 neurons in the first hidden layer and 3 in the second one
    clf = MLPClassifier(solver="sgd", alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
    clf.fit(X_train, y_train)

    # Use the predict_proba() function to return the estimated probability of three classes
    prediction_probability = clf.predict_proba(X_test[2:3,:])
    print("\nPrediction Probability: ", prediction_probability)

    # Use the predict() function to predict which class the data points belong to
    prediction_result = clf.predict(X_test[2:3,:])
    print("\nPrediction Result: ", prediction_result) # The prediction result is class 2

    # Use the score() to compute the mean accuracy of the given test data and labels
    acc = clf.score(X_test, y_test)
    print("\nMean Accuracy: ", acc)

if __name__ == "__main__":
    main()