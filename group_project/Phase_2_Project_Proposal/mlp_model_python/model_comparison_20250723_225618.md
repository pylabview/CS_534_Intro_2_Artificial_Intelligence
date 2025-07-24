# Machine‑Failure Model Comparison

## Table 1 – 5‑fold CV Results

| Model              | Best Params                                                                                                               |   CV MCC |
|:-------------------|:--------------------------------------------------------------------------------------------------------------------------|---------:|
| SVM                | {'clf__C': 100, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}                                                              | 0.812246 |
| DecisionTree       | {'clf__ccp_alpha': 0.01, 'clf__criterion': 'entropy', 'clf__max_depth': 5}                                                | 0.807628 |
| KNN                | {'clf__algorithm': 'auto', 'clf__n_neighbors': 5, 'clf__p': 2}                                                            | 0.739155 |
| LogisticRegression | {'clf__C': 10, 'clf__penalty': 'l2'}                                                                                      | 0.63627  |
| MLP                | {'clf__learning_rate': 'adaptive', 'clf__hidden_layer_sizes': (100, 50), 'clf__alpha': 0.0001, 'clf__activation': 'tanh'} | 0.603964 |

## Table 2 – 20 % Hold‑out Test Results

| Model              | Best Params                                                                                                                                                                   |   Test MCC |
|:-------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|
| DecisionTree       | {'memory': None, 'steps': [('scale', ColumnTransformer(transformers=[('scale', StandardScaler(),                                                                              |   0.823886 |
|                    |                                  ['Air temperature [K]',                                                                                                                      |            |
|                    |                                   'Process temperature [K]',                                                                                                                  |            |
|                    |                                   'Rotational speed [rpm]', 'Torque [Nm]',                                                                                                    |            |
|                    |                                   'Tool wear [min]'])])), ('clf', DecisionTreeClassifier(ccp_alpha=0.01, criterion='entropy', max_depth=5,                                    |            |
|                    |                        random_state=42))], 'transform_input': None, 'verbose': False}                                                                                         |            |
| KNN                | {'memory': None, 'steps': [('scale', ColumnTransformer(transformers=[('scale', StandardScaler(),                                                                              |   0.823529 |
|                    |                                  ['Air temperature [K]',                                                                                                                      |            |
|                    |                                   'Process temperature [K]',                                                                                                                  |            |
|                    |                                   'Rotational speed [rpm]', 'Torque [Nm]',                                                                                                    |            |
|                    |                                   'Tool wear [min]'])])), ('clf', KNeighborsClassifier())], 'transform_input': None, 'verbose': False}                                        |            |
| SVM                | {'memory': None, 'steps': [('scale', ColumnTransformer(transformers=[('scale', StandardScaler(),                                                                              |   0.808911 |
|                    |                                  ['Air temperature [K]',                                                                                                                      |            |
|                    |                                   'Process temperature [K]',                                                                                                                  |            |
|                    |                                   'Rotational speed [rpm]', 'Torque [Nm]',                                                                                                    |            |
|                    |                                   'Tool wear [min]'])])), ('clf', SVC(C=100, probability=True, random_state=42))], 'transform_input': None, 'verbose': False}                 |            |
| MLP                | {'memory': None, 'steps': [('scale', ColumnTransformer(transformers=[('scale', StandardScaler(),                                                                              |   0.693053 |
|                    |                                  ['Air temperature [K]',                                                                                                                      |            |
|                    |                                   'Process temperature [K]',                                                                                                                  |            |
|                    |                                   'Rotational speed [rpm]', 'Torque [Nm]',                                                                                                    |            |
|                    |                                   'Tool wear [min]'])])), ('clf', MLPClassifier(activation='tanh', early_stopping=True,                                                       |            |
|                    |               hidden_layer_sizes=(100, 50), learning_rate='adaptive',                                                                                                         |            |
|                    |               max_iter=400, random_state=42))], 'transform_input': None, 'verbose': False}                                                                                    |            |
| LogisticRegression | {'memory': None, 'steps': [('scale', ColumnTransformer(transformers=[('scale', StandardScaler(),                                                                              |   0.634069 |
|                    |                                  ['Air temperature [K]',                                                                                                                      |            |
|                    |                                   'Process temperature [K]',                                                                                                                  |            |
|                    |                                   'Rotational speed [rpm]', 'Torque [Nm]',                                                                                                    |            |
|                    |                                   'Tool wear [min]'])])), ('clf', LogisticRegression(C=10, random_state=42, solver='liblinear'))], 'transform_input': None, 'verbose': False} |            |

### Conclusion

The **DecisionTree** model achieved the highest MCC on the test set and is selected for deployment.
