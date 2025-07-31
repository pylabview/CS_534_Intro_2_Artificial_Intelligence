import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,roc_curve,auc, precision_recall_curve,average_precision_score
import joblib


#Visually shows statistics of FPs, FNs, TPs, TNs
def confusion_bar(cm, title, colors, save_path= None):
    tn, fp, fn, tp = cm.ravel()
    categories = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    values = [tn, fp, fn, tp]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, values, color=colors)
    plt.title(title, color = "black")
    plt.ylabel('Count', color = "black")
    plt.ylim(0, max(values) * 1.1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + max(values) * 0.02, int(yval), ha='center', va='bottom', color="black")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_CR(y_true, y_pred,filename):
    report = classification_report(y_true, y_pred, digits=6)
    with open(filename, "w") as f:
        f.write(report)


#ROC Curve
def plot_roc_pr_curves(y_true, y_scores, model_name):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='black')
    plt.ylabel('True Positive Rate', color='black')
    plt.title(f'{model_name} ROC Curve (Test Set)', color='black')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'{model_name.lower()}_roc_curve_test_set.png')
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', color='black')
    plt.ylabel('Precision', color='black')
    plt.title(f'{model_name} Precision-Recall Curve (Test Set)', color='black')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'{model_name.lower()}_pr_curve_test_set.png')
    plt.show()

def main():

    #Load test.csv data and split into X and y
    test_df = pd.read_csv('test.csv')
    y_test = test_df['label']
    X_test = test_df.drop(columns=['label'])

    #Load my trained and saved Random Forest model and print the best parameters
    best_rf = joblib.load('Final_random_forest_model.pkl')
    print("Random Forest Parameters:")
    print(best_rf.get_params())

    #Generate predictions of the test set using the best Random Forest Model
    rf_pred = best_rf.predict(X_test)

    #Generate probabilities for the ROC and Precision-Recall curve
    rf_prob = best_rf.predict_proba(X_test)[:, 1]

    #Print accuracy of Random forest model on test data
    print("Random Forest Test Accuracy:", accuracy_score(y_test, rf_pred))

    #Visuallize confusion bar
    confusion_bar(confusion_matrix(y_test,rf_pred),
                  'Random Forest Confusion Matrix (Test Set)',
                  colors =["navy", "gold", "gold", "navy"],
                  save_path="rf_confusion_matrix_final.png")

    #Save classification report to a new file
    save_CR(y_test, rf_pred, "rf_CR_test_set.txt")

    # Plot and display ROC and Precision-Recall curves for Random Forest
    plot_roc_pr_curves(y_test, rf_prob, "Random Forest")

    # Load my trained and saved XG Boost model and print the best parameters
    best_xgb = joblib.load('Final_xgboost_model.pkl')
    print("XGBoost Parameters:")
    print(best_xgb.get_params())

    # Generate predictions of the test set using the best XG Boost Model
    xgb_pred = best_xgb.predict(X_test)

    # Generate probabilities for the ROC and Precision-Recall curve
    xgb_prob = best_xgb.predict_proba(X_test)[:, 1]

    # Print accuracy of XG Boost model on test data
    print("XGBoost Test Accuracy:", accuracy_score(y_test, xgb_pred))

    # Visuallize confusion bar
    confusion_bar(confusion_matrix(y_test,xgb_pred),
                  'XGBoost Confusion Matrix (Test Set)',
                  colors =["darkgreen", "orangered", "orangered", "darkgreen"],
                  save_path="xgboost_confusion_matrix_final.png")

    # Save classification report to a new file
    save_CR(y_test, xgb_pred, "xgboost_CR_test_set.txt")

    # Plot and display ROC and Precision-Recall curves for XGBoost
    plot_roc_pr_curves(y_test, xgb_prob, "XGBoost")

if __name__ == '__main__':
    main()
