import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def main():
    df = pd.read_csv('longley.csv', index_col=0)
    print("Correlation coefficient = ", np.corrcoef(df.Employed,df.GNP)[0,1])

    X = df.Employed  # predictor (independent variable)
    y = df.GNP  # response (dependent variable)
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    lr_model = sm.OLS(y, X).fit()
    print(lr_model.summary())

    # We pick 100 points equally spaced from the min to the max
    X_prime = np.linspace(X.Employed.min(), X.Employed.max(), 100)
    X_prime = sm.add_constant(X_prime)  # Add a constant as we did before

    # Now we calculate the predicted values
    y_hat = lr_model.predict(X_prime)

    plt.scatter(X.Employed, y)  # Plot the raw data
    plt.xlabel("Total Employment")
    plt.ylabel("Gross National Product")
    plt.plot(X_prime[:, 1], y_hat, 'red', alpha=0.9)  # Add the regression line, colored in red
    plt.show()

if __name__ == "__main__":
    main()