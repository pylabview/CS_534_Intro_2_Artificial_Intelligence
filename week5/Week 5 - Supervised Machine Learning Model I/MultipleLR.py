import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D

def main():
    # Load the advertising dataset into a pandas data frame
    df = pd.read_csv('Advertising.csv', index_col=0)
    y = df['Sales']
    X = df[['TV','Radio']]
    X = sm.add_constant(X)

    lr_model = sm.OLS(y,X).fit()
    print(lr_model.summary())
    print(lr_model.params)

    # Figure out X and Y axis using ranges from TV and Radio
    X_axis, Y_axis = np.meshgrid(np.linspace(X.TV.min(), X.TV.max(), 100), np.linspace(X.Radio.min(), X.Radio.max(), 100))

    # Plot the hyperplane by calculating corresponding Z axis (Sales)
    Z_axis = lr_model.params[0] + lr_model.params[1] * X_axis + lr_model.params[2] * Y_axis

    # Create matplotlib 3D axes
    fig = plt.figure(figsize=(12, 8))   # figsize refers to width and height of the figure
    ax = Axes3D(fig, azim=-100)

    # Plot hyperplane
    ax.plot_surface(X_axis, Y_axis, Z_axis, cmap=plt.cm.coolwarm, alpha=0.5, linewidth=0)

    # Plot data points
    ax.scatter(X.TV, X.Radio, y)

    # Set axis labels
    ax.set_xlabel('TV')
    ax.set_ylabel('Radio')
    ax.set_zlabel('Sales')

    plt.show()

if __name__ == "__main__":
    main()