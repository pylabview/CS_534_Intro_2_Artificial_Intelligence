import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def main():
    t_df = pd.read_csv('titanic_data.csv', index_col='PassengerId')
    t_df = t_df.dropna()
    t_df.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
    t_df['Sex'].replace(['male', 'female'], [1, 0], inplace=True)
    t_df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    X = t_df.drop(columns=['Survived'])
    y = t_df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    logmodel = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=False)
    print(logmodel.summary())

    # Form our predictions, convert continuous [0, 1] predictions to binary
    predictions = logmodel.predict(sm.add_constant(X_test))
    bin_predictions = [1 if x >= 0.5 else 0 for x in predictions]

    # We can now assess the accuracy and print out the confusion matrix
    print(accuracy_score(y_test, bin_predictions))
    print(confusion_matrix(y_test, bin_predictions))

    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.3f)' % roc_auc)
    plt.title('ROC Curve (area = %0.3f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

if __name__ == "__main__":
    main()