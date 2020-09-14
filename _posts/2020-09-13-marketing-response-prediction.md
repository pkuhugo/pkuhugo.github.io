---
title: "Marketing Customer Response Prediction"
date: 2020-09-13
tags: [machine learning]
header:
  image: "/images/05_marketing/00_cover.png"
excerpt: "Machine Learning"
mathjax: "true"
---

# Introduction
n this project, I am going to build a model to predict whether a customer will respond to the marketing campaign based on the training dataset. I'll use the model to predict the customer responses using the features provided in the test dataset.

In the training dataset, there are 7414 samples and 21 predictor feature. The target variable "responded" has two values, "yes" and "no". Thus, this project is a binary classification problem. In the exploratory analysis, the 21 predictors were separated into three categories: personal, bank activity ,and macro-economic data, and their statistics were examined to get some intuitive between the predictors and target.

In the data selection, the three highly correlated variables were dropped to improve the model (linear models) stability. The categorical features were converted to numerical data as the input for the model training. Before model training, the number of samples in the minority class was balanced using Synthetic Minority Oversampling Technique.

Three different algorithms were adopted for model training, logistic regression, random forest, and gradient boosting tree (XGboost). The model performance was evaluated based on the ROC_AUC score. The random forest and gradient boosting tree models' average performance were similar with ROC_AUC score near 0.8, but on the class responded with 'Yes' value, the gradient boosting tree model is slightly better. In the three models, the model performance was not so great in class responded with 'Yes'. The best recall score is only 0.23, which means only 23% of potential customers were correctly predicted, even though the weighted average recall is 0.88. There's still some potential to fine-tune the model to increase the prediction result in class 'Yes'.

The top important factors of predicting customer response are also identified based on the importance analysis. The strong indicators of the customer response are previous campaign outcome, previous contact, macro-economic, campaign month, and contact type.".

The assignment was intended to build a model that can be used to create some useful results rather than to obtain the finest model. To improve the prediction results, some experiments are worth testing:

Feature engineering to extract useful information from current predictors.
Test class weight parameters to improve the results in class responded "Yes".
Other models can also be tested, for example, Neural network.

# Exploratory Data Analysis
## Plot the Statistics of Predictors in the Training Datasets
The features are classified to three categories:
1. customer data: custage, profession, marital, schooling, default, housing, loan
2. related with bank activity: cantact, month, day_of_week, compaign, pdays, pmonth, previous, poutcome, pastemail,
3. social and economic factors: emp.var.rate, cons.price.idx, cons.conf.idx, eribor3m, nr.employed

Customer Age:
![alt]({{ site.url }}{{ site.baseurl }}/images/05_marketing/04_age.jpg)

Previous Outcome:
![alt]({{ site.url }}{{ site.baseurl }}/images/05_marketing/01_poutcome.jpg)

Macro Economic Index:
![alt]({{ site.url }}{{ site.baseurl }}/images/05_marketing/02_rate.jpg)

Campaign Month:
![alt]({{ site.url }}{{ site.baseurl }}/images/05_marketing/03_month.jpg)

By plotting the statistics of the predictors in the training dataset, I gained some intuitive thoughts of the relationship between the predictor and target variables. For some features, the distributions in the response and non-response customers are indistinguishable, which made them potentially not strong predictors, like customer age. On the other hand, some predictors show different values for different type of customers, for instance, previous campain outcome, the European 3 months rate, and the campaign conduct time.  

## Remove the Highly Correlated Features
We found highly correlated features (correlation coefficient > 0.9):

pdays and pmonths
emp.var.rate and euribor3m
euribor3m and nr.employed
Thus, we are going to drop "pmomths", "emp.var.rate", "nr.employed" from the predictor features in order to get more stable models.

Campaign Month:
![alt]({{ site.url }}{{ site.baseurl }}/images/05_marketing/05_corr.jpg)

## Handling Missing Data
In both train and test datasets, three features contain missing data: custAge, schooling, and day_of_week. Among these three features, custAge is a numeric feature, while schooling and day_of_week are categorical features. By plotting the statistics of the custAge, I did not observe a strong relationship between the feature and the target value, thus it's acceptable to impute the missing data with the median value of custAge, which is 38. For the categorical features with missing data, I replaced the missing data with an additional category "missing_value", which is similar to the "unknown" value in those features.

Python code block:
```python
# Impute the missing value for both numerical features
imputer = SimpleImputer(strategy = 'median')
imputer.fit(df_train_feature[num_feature_list])
df_train_feature[num_feature_list] = pd.DataFrame(imputer.transform(df_train_feature[num_feature_list]), columns = num_feature_list)
df_test_feature[num_feature_list] = pd.DataFrame(imputer.transform(df_test_feature[num_feature_list]), columns = num_feature_list)

# Impute the missing value for both categorical features
imputer = SimpleImputer(strategy = 'constant')
imputer.fit(df_train_feature[cat_feature_list])
df_train_feature[cat_feature_list] = pd.DataFrame(imputer.transform(df_train_feature[cat_feature_list]), columns = cat_feature_list)
df_test_feature[cat_feature_list] = pd.DataFrame(imputer.transform(df_test_feature[cat_feature_list]), columns = cat_feature_list)
```

## Handle Categorical Data
I used one-hot-encoding to convert categorical data to numerical data which can be used in machine learning algorithms. In this assignment, almost all categorical data has no specific order related to the categorical values, thus it's reasonable to use one-hot-encoding. If there's inherent order associated with values in the categorical feature, label encoding may be used.
Python code block:
```python
#Use one-hot-code to convert categorical data to numerical data
df_cmb = pd.concat((df_train_feature, df_test_feature), axis = 0)

print(df_cmb.shape)
for f in dummy_list:
    dummy_df = pd.get_dummies(df_cmb[f], prefix = f)
    df_cmb = df_cmb.drop([f], axis=1)
    df_cmb = pd.concat((df_cmb, dummy_df), axis = 1)
print(df_cmb.shape)
```


## Handle Unbalanced Data
In this assignment, the number of data with target value 0 is about 8 times the data with target value 1. I used oversampling based on Synthetic Minority Oversampling Technique (SMOTE). A random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space.

```python
# The ratio of target value 0 to target value 1 is about 8.
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
```

# Model Training and Model Selection
The training data was split into two parts: training and validation with a ratio of 8:2. The models are training using K-fold methods to avoid overfitting. The parameters were tuned based on grid search or a Bayesian-based hyper-parameter tuning method. The ROC AUC score was used to evaluate the model since the imbalance of the sample numbers in a different class.

I choose the gradient boosting tree model as the final model to predict based on the test data. The gradient boosting tree model is tree-based model and was build based on an ensemble of many individual trees. Each tree estimator was built sequentially that the next tree was build to minimize the residual after the current tree. The model and residual was updated with a weighted factor after each step.

In this assignment, the gradient boosting tree provided the best score in both training and validation data. The way gradient boosting tree builds its final model was able to extract useful information from the weak predictors. Though a lot of the time, the training of gradient boosting tree model is slow. The data size is small in this case, thus the training cost is not an issue.

I have tried another two models, logistic regression and random forest. The logistic regression training is pretty fast, but the result score is the lowest among the three. logistic regression is more stable than tree based, but the trade off is the accuracy is not as good as the tree based model in this case. The predictors in the training dataset also have a lot of missing or unknown value which may cause problems in linear regression models but not in tree based models.

The random forest model actually provided very close result as the gradient boosting tree model. Both are tree based models, but in random forest, the trees are built simultaneously with subsample of the training sample and predictors, while the trees are built sequentially to target minimizing the residuals. Usually, the gradient boosting tree result in training dataset is better, but likely overfitting. Proper parameter tuning is needed to avoid the problem. In the end, in class responded "yes", the gradient boosting tree result is slightly better.

```python
space ={
        'n_estimators' : hp.choice('n_estimators', np.arange(800, 1500, 100, dtype=int)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
        'max_depth': hp.choice("max_depth", np.arange(1, 3, 1, dtype=int)),
        'min_child_weight' : hp.choice("min_child_weight", np.arange(2, 6, 2, dtype=int)),
        'subsample': hp.quniform('subsample', 0.7, 1.0, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
    }

def objective(space):

    classifier = xgb.XGBClassifier(n_estimators = int(space['n_estimators']),
                            max_depth = int(space['max_depth']),
                            learning_rate = space['learning_rate'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree'],
                            tree_method='gpu_hist', gpu_id=0, nthread=-1)
    # Applying k-Fold Cross Validation   
    scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, scoring = 'roc_auc', cv = 3)
    CrossValMean = scores.mean()

    print("CrossValMean:", CrossValMean)

    return{'loss':1-CrossValMean, 'status': STATUS_OK }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            return_argmin=False,
            trials=trials)

print("Best: ", best)
```
ROC of different models:
![alt]({{ site.url }}{{ site.baseurl }}/images/05_marketing/06_roc.jpg)

Now, we have to decide which model is the best model, and we have two types of wrong values:<br>
1. False positive, means the customers did not response to the campaign, but the model thinks they did
2. False negative, means the customer did response to the campaign, but the model thinks they did not

I think the second is most harmful, because we may miss the target customers who are most likely to buy the insurance in the feature campaign. The first error is ok, since the worst thing is to spend some extra money, but we won't loose potential customers.

Confusion Matrix of Gradient Boosting Tree Result:
![alt]({{ site.url }}{{ site.baseurl }}/images/05_marketing/07_metrics.jpg)

The random forest and gradient boosting tree models average performance was similar with ROC AUC score near 0.8, but on the class responded with 'Yes' value, the gradient boosting tree model is slightly better. In the three models, the model performance was not so great in class responded with 'Yes'. The best recall score is only 0.23, which means only 23% potential customers were correctly predicted, even though the weighted average recall is 0.88. There's still some potential to fine tune the model to increase the prediction result in class 'Yes'. The top important factors of predicting customer response are also identified based on the importance analysis. The strong indicators of the customer response are: previous campaign outcome, previous contact, macro economic, campaign month, and contact type.

# Future Work
The project was intended to build a model that can be used to create some useful result rather than to obtain the best model. In order to improve the prediction results, some experiments are worth testing:

Feature engineering to extract useful information from current predictors.
Test class weight parameters to improve the results on class responded "Yes".
Other models can also be tested, for example Neural network.
