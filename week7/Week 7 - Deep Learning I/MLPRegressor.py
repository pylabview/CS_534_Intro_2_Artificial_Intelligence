import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

def main():
    # Load our dataset
    raw_data = "yacht_hydrodynamics.data"
    dataset = np.loadtxt(raw_data)
    print("Dataset:")
    print(dataset)
    print("\nShape of Dataset: ", dataset.shape) # We have 308 data points and the 7 features listed above.

    # Use the StandardScaler() function to normalize the dataset
    data_scaled = StandardScaler().fit_transform(dataset)

    # Create partitions of training and testing
    # The first six columns are features and the 7th column is the target
    X_train, X_test, y_train, y_test = train_test_split(data_scaled[:, 0:6], data_scaled[:, 6],
                                                        test_size=0.25, random_state=1)

    # 3 neurons in the first hidden layer and five in the second one
    regr = MLPRegressor(hidden_layer_sizes=(3, 5), random_state=1).fit(X_train, y_train)

    # Use the predict() function with the test data to predict the values of the target variable
    print("*", X_test[2:3,:])
    prediction_result = regr.predict(X_test[2:3,:])
    print("\nPrediction Result: ", prediction_result)

    # Use the score() to compute the squared error of our model
    error = regr.score(X_test, y_test)/len(y_test)
    print("\nSquared Error: ", error)

if __name__ == "__main__":
    main()