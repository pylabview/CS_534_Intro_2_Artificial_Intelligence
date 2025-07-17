import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split

def main():
    bank_df = pd.read_csv('bank.csv', delimiter=';')
    for col in bank_df.columns:
        labels, uniques = pd.factorize(bank_df[col])
        bank_df[col] = labels

    y = bank_df['y']
    X = bank_df.drop(columns='y')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Multinomial Naïve Bayes assumes categories follow multinomial distribution
    nb_multi = MultinomialNB()
    nb_multi.fit(X, y)

    # Gaussian Naïve Bayes assumes categories follow normal distribution
    nb_gauss = GaussianNB()
    nb_gauss.fit(X, y)

    from sklearn.metrics import accuracy_score, confusion_matrix
    multi_preds = nb_multi.predict(X_test)
    print("Results for multinomial distribution assumption:")
    print(accuracy_score(y_test, multi_preds))
    print(confusion_matrix(y_test, multi_preds))

    print()
    print("Results for Gaussian distribution assumption:")
    gauss_preds = nb_gauss.predict(X_test)
    print(accuracy_score(y_test, gauss_preds))
    print(confusion_matrix(y_test, gauss_preds))

if __name__ == "__main__":
    main()