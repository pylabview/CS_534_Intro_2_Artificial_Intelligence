import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
import graphviz

def main():
    bal_df = pd.read_csv('balloons.csv')

    # convert categorical data to int representations of unique categories
    for col in bal_df.columns:
        labels, uniques = pd.factorize(bal_df[col])
        bal_df[col] = labels

    X = bal_df.drop(columns='inflated')
    y = bal_df['inflated']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    dot_data = tree.export_graphviz(dtree, out_file=None,
                                    feature_names=('Color', 'size', 'act', 'age'),
                                    class_names=('0','1'),
                                    filled=True)
    graph = graphviz.Source(dot_data, format="png")
    graph.render('balloons_dt', view=True)

if __name__ == "__main__":
    main()