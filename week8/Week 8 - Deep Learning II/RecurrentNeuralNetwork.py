import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
import tensorflow as tf
from keras.models import Sequential

bikes = []

def data_bike_num(path_to_dataset='bike_rnn.csv',
                           sequence_length=20,
                           ratio=1.0):
    max_values = ratio * 45949
    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=",")
        next(data, None)  # skip the headers
        nb_of_values = 0
        for line in data:
            try:
                bikes.append(float(line[0]))
                nb_of_values += 1
            except ValueError:
                pass
            if nb_of_values >= max_values:
                break
    print ("Data loaded from csv. Formatting...")
    print("Number of Records: ", len(bikes))
    print()
    
    result = []
    for index in range(len(bikes) - sequence_length):
        result.append(bikes[index: index + sequence_length])
    result = np.array(result)  # shape (2049230, 50)
    result_mean = result.mean()
    result -= result_mean
    print("Shift: ", result_mean)
    print ("Data: ", result.shape)
    row = int(round(0.95 * result.shape[0]))
    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1] # 2297
    y_test = result[row:, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return [X_train, y_train, X_test, y_test, result_mean]

# Function to build the RNN-based model
# Return sequences refer to return the hidden state a<t>.
# By default, the return_sequences is set to False in Keras RNN layers, and
# this means the RNN layer will only return the last hidden state output a<T>.
# The last hidden state output captures an abstract representation of the input sequence.
# When return_sequences parameter is True, it will output all the hidden states of each time steps.
def build_model():
    model = Sequential()
    layers = [1, 50, 100, 1]

    # layers[1]: 50 neurons in this SimpleRNN layer.
    # input_shape=(None, layers[0]): A None dimension means that it can be any scalar number.
    # layers[0] = 1 refers to 1 dimensional input vector.
    # return_sequences=True: It will output all the hidden states of each time steps.

    model.add(tf.keras.layers.SimpleRNN(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.SimpleRNN(
        layers[2],
        return_sequences=False)) # This means the RNN layer will only return the last hidden state output
    model.add(Dropout(0.2))
    model.add(Dense(layers[3])) # This output layer only has 1 neuron
    model.add(Activation("linear"))
    start = time.time() # time.time() method of Time module is used to get the time in seconds since epoch.
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mae', 'mape'])
    print("Compilation Time : ", time.time() - start)
    return model

def main():
    epochs = 2
    ratio = 1
    sequence_length = 20
    path_to_dataset = 'bike_rnn.csv'
    X_train, y_train, X_test, y_test, result_mean = data_bike_num(path_to_dataset, sequence_length, ratio)

    print ('\nData Loaded. Compiling...\n')
    model = build_model()
    model.fit(
        X_train, y_train,
        batch_size=512, epochs=epochs, validation_split=0.05)
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))

    # Evaluate
    # MSE = Mean Squared Error
    # MAE = Mean Absolute Error
    # MAPE = Mean Absolute Percentage Error
    scores = model.evaluate(X_test, y_test, batch_size=512)
    print("\nEvaluation results: \nMSE={:.6f}\nMAE={:.6f}\nMAPE={:.6f}".format(scores[0], scores[1], scores[2]))

    # Draw the figure
    y_test += result_mean
    predicted += result_mean
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test,label="Real")
    ax.legend(loc='upper left')
    plt.plot(predicted,label="Prediction")
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()