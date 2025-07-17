import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("wine.csv")

    # Mark about 70% of the data for training and use the rest for
    # testing
    # We will use 'density', 'sulphates', and 'residual_sugar'
    # features for training a classifier on 'high_quality'
    X_train, X_test, y_train, y_test = train_test_split(df[['density','sulphates','residual_sugar']], 
                                                        df['quality'], test_size=.3)

    # Define the classifier using kNN function and train it
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    # Test the classifier by giving it test instances
    prediction = classifier.predict(X_test)

    # Count how many were correctly classified
    correct = np.where(prediction==y_test, 1, 0).sum()
    print(correct)

    # Calculate the accuracy of this classifier
    accuracy = correct/len(y_test)
    print(accuracy)

    # Start with an array where the results (k and corresponding
    # accuracy) will be stored
    results = []

    for k in range(1, 51):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        accuracy = np.where(prediction==y_test, 1, 0).sum() / (len(y_test))
        print ("k=",k," Accuracy=", accuracy)
        results.append([k, accuracy]) # Storing the k,accuracy tuple in results array

    # Convert that series of tuples in a dataframe for easy plotting
    results = pd.DataFrame(results, columns=["k", "accuracy"])

    plt.plot(results.k, results.accuracy)
    plt.title("Value of k and corresponding classification accuracy")
    plt.show()

if __name__ == "__main__":
    main()