# Machine-Failure Model Comparison

## Table 1 – 5-fold CV (80% train)

| Model              | Best Params                                                                                                              |   CV MCC (5-fold) |
|:-------------------|:-------------------------------------------------------------------------------------------------------------------------|------------------:|
| MLP                | {'clf__learning_rate': 'constant', 'clf__hidden_layer_sizes': (100, 50), 'clf__alpha': 0.001, 'clf__activation': 'relu'} |            0.8232 |
| SVM                | {'clf__kernel': 'rbf', 'clf__gamma': 'auto', 'clf__C': 100}                                                              |            0.8196 |
| KNN                | {'clf__p': 2, 'clf__n_neighbors': 7, 'clf__algorithm': 'ball_tree'}                                                      |            0.7539 |
| DecisionTree       | {'clf__ccp_alpha': 0.0, 'clf__criterion': 'entropy', 'clf__max_depth': 5}                                                |            0.8052 |
| LogisticRegression | {'clf__C': 10, 'clf__penalty': 'l2'}                                                                                     |            0.632  |

## Table 2 – Hold-out Test (20%)

| Model              | Best Params                                                                                                              |   Test MCC (20%) |
|:-------------------|:-------------------------------------------------------------------------------------------------------------------------|-----------------:|
| MLP                | {'clf__learning_rate': 'constant', 'clf__hidden_layer_sizes': (100, 50), 'clf__alpha': 0.001, 'clf__activation': 'relu'} |           0.8239 |
| SVM                | {'clf__kernel': 'rbf', 'clf__gamma': 'auto', 'clf__C': 100}                                                              |           0.7945 |
| KNN                | {'clf__p': 2, 'clf__n_neighbors': 7, 'clf__algorithm': 'ball_tree'}                                                      |           0.7945 |
| DecisionTree       | {'clf__ccp_alpha': 0.0, 'clf__criterion': 'entropy', 'clf__max_depth': 5}                                                |           0.747  |
| LogisticRegression | {'clf__C': 10, 'clf__penalty': 'l2'}                                                                                     |           0.6062 |

### Conclusion
The **MLP** achieved the highest MCC on both CV and the test set, so it is selected as the production model.
