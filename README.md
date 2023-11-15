
#  Predicting Employee Access Approval in Amazon.com

## Introduction:
In today's fast-paced business environment, the effective management of employee access to organizational resources is crucial for maintaining security and operational efficiency. Predicting whether access requests will be approved or denied is a complex task that can benefit from advanced machine learning algorithms. This report delves into the development of a predictive model using the CatBoost algorithm to forecast access outcomes for unseen employees in the context of Amazon.com. The dataset encompasses various features such as RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2, ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, and ROLE_CODE.

## Steps:

### 1. Data Overview:
The dataset at hand is comprised of two main sets: a training set with 32,769 entries and a test set with 58,921 entries. The training set includes a crucial target variable, denoted as "ACTION," indicating whether access was approved (1) or denied (0). The features in both sets encompass diverse attributes related to roles, departments, and individual employees.

### 2. Data Exploration:
#### Data Imbalance:
```python
# Display the distribution of target variable 'ACTION'
traindf['ACTION'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of Access Approval (1) and Denial (0)')
plt.xlabel('Access Outcome')
plt.ylabel('Count')
plt.show()
```

#### Feature Distribution:
```python
# Visualize the distribution of selected features
for feature in traindf.columns[1:]:
    sns.distplot(traindf[feature].dropna())
    plt.title(f'Distribution of {feature}')
    plt.show()
```

### 3. Data Preprocessing:
#### Categorical Features:
```python
# Identify categorical features
categorical_features = list(range(x.shape[1]))

# Train CatBoost with categorical feature specification
catmodel = CatBoostClassifier(cat_features=categorical_features, verbose=200, random_seed=1)
catmodel.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
```

#### Train-Test Split:
To facilitate model evaluation, the dataset is judiciously split into training and validation sets.

### 4. Model Building:
#### CatBoost Classifier:
```python
# CatBoost model without explicit categorical feature specification
catmodel_1 = CatBoostClassifier(eval_metric='AUC', verbose=200, random_seed=1)
catmodel_1.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)

# CatBoost model with categorical features explicitly defined
catmodel = CatBoostClassifier(cat_features=categorical_features, eval_metric='AUC', verbose=200, random_seed=1)
catmodel.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
```

#### Hyperparameter Tuning:
Model training encompasses hyperparameter tuning, and the optimal model is selected based on its performance on the validation set.

### 5. Feature Importance:
```python
# Plot feature importance
feature_imp = catmodel.get_feature_importance(prettified=True)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importances', y='Feature Id', data=feature_imp)
plt.title('CatBoost Features Importance')
plt.show()
```

## Source:
The analytical procedures are executed using the Python programming language, employing prominent data science libraries such as pandas, seaborn, and CatBoost. The CatBoost algorithm is specifically chosen for its native support for categorical features, streamlining the model development process.


### Model Evaluation and Validation:
```python
# Assess model performance on the validation set
validation_accuracy = catmodel.score(x_valid, y_valid)
print(f'Model Validation Accuracy: {validation_accuracy}')

# Generate predictions on the test set
predictions = catmodel.predict(x_test)
```

## Conclusion:
In conclusion, the developed CatBoost model showcases commendable performance in predicting employee access approval. The insights gained from feature importance analysis provide valuable information for decision-makers. While the model holds promise, ongoing monitoring and periodic updates are imperative to adapt to evolving access patterns, organizational changes, and emerging security threats. Implementing this predictive model can contribute to the automation of access approval processes, enhancing organizational efficiency, and bolstering cybersecurity measures. As machine learning continues to advance, the integration of predictive analytics in access management becomes an increasingly strategic endeavor for modern enterprises.
