---
title: "Lithology Facies Classification"
date: 2020-08-14
tags: [machine learning]
header:
  image: "/images/02_well/profiles_cut.jpg "
excerpt: "Machine Learning", "Feature Engineering"
mathjax: "true"
---

# Introduction
In this project, I am going to demonstrate using the machine learning algorithms to identify lithofacies based on well-log measurements. A supervised-learning algorithm Gradient Boosting Tree is trained by 9 well-log data. These wells have already had lithofacies classes assigned. Once the classifier is trained, I'll use it to assign facies to wells that have not been described.

# Data Description
The demo data set was from University of Kansas and they were collected on the Hugoton and Panoma gas fields. For more on the origin of the data, see Dubois et al. (2007) and the Jupyter notebook that accompanies this tutorial at [http://github.com/seg](http://github.com/seg).

The data set included 9 wells data, and there were 3232 observations. There are 9 columns in the data table, including 5 wireline log measurements, 2 indicator variables derived from geologic knowledge, a facies label at half foot intervals, and a relative position.

The seven predictor variables are:
* Five wire line log curves include [gamma ray](http://petrowiki.org/Gamma_ray_logs) (GR), [resistivity logging](http://petrowiki.org/Resistivity_and_spontaneous_%28SP%29_logging) (ILD_log10),
[photoelectric effect](http://www.glossary.oilfield.slb.com/en/Terms/p/photoelectric_effect.aspx) (PE), [neutron-density porosity difference and average neutron-density porosity](http://petrowiki.org/Neutron_porosity_logs) (DeltaPHI and PHIND). Note, some wells do not have PE.
* Two geologic constraining variables: nonmarine-marine indicator (NM_M) and relative position (RELPOS)

The nine discrete facies (classes of rocks) are:
1. Nonmarine sandstone
2. Nonmarine coarse siltstone
3. Nonmarine fine siltstone
4. Marine siltstone and shale
5. Mudstone (limestone)
6. Wackestone (limestone)
7. Dolomite
8. Packstone-grainstone (limestone)
9. Phylloid-algal bafflestone (limestone)

These facies aren't discrete, and gradually blend into one another. Some have neighboring facies that are rather close.  Mislabeling within these neighboring facies can be expected to occur.  The following table lists the facies, their abbreviated labels and their approximate neighbors.

Facies |Label| Adjacent Facies
:---: | :---: |:--:
1 |SS| 2
2 |CSiS| 1,3
3 |FSiS| 2
4 |SiSh| 5
5 |MS| 4,6
6 |WS| 5,7
7 |D| 6,8
8 |PS| 6,7,9
9 |BS| 7,8

Here's the depth profiles for the fine well-log measurements and the corresponding facies label:
![alt]({{ site.url }}{{ site.baseurl }}/images/04_facies/01_profile.png)

Distribution of the training data by Facies:
![alt]({{ site.url }}{{ site.baseurl }}/images/04_facies/02_distribution.png)

The crossplots are used to visualize how tow features vary with rock type. We created a matrix of crossplots to visualize the variation between log measurements in the data set.
![alt]({{ site.url }}{{ site.baseurl }}/images/04_facies/03_crossplot.png)

# Feature engineering
The relationship between the well-log features and the lithofacies is complicated. In the training dataset, we only have 5 well-log features and 2 derived features. To reveal the underline relationship, we proposed the following feature augmentation.
1. neighboring value
2. gradient
3. quadratic expansion
After the Feature Engineering, the total number of features increased to 435.
```python
# Feature windows concatenation function
def augment_features_window(X, N_neig):

    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):

    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff

    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))

    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):

    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)

    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])

    return X_aug, padded_rows

X_aug, padded_rows = augment_features(X, well, depth)
```
Quadratic expansion
```python
deg = 2
poly = preprocessing.PolynomialFeatures(deg, interaction_only=False)
X_aug2 = poly.fit_transform(X_aug)
```

# Building Machine Learning Model
To evaluate the accuracy of the classifier, we will remove one well from the training set. We will compare the predicted facies with the pre-labeled value.
```python
#leave a well for blind test later
blind = X_aug2[training_data['Well Name'] == 'CHURCHMAN BIBLE']
blind_y = y[training_data['Well Name'] == 'CHURCHMAN BIBLE']
X_aug_tr = X_aug2[training_data['Well Name'] != 'CHURCHMAN BIBLE']
y_tr = y[training_data['Well Name'] != 'CHURCHMAN BIBLE']
```

Many machine-learning algorithms assume the feature data are normally distributed (i.e., Gaussian distributed with zero mean and unit variance). We will apply the python package SandardScaler to the predictor features, thus they'll have the property.
```python
scaler = preprocessing.StandardScaler().fit(X_aug_tr)
X_aug_scaled = scaler.transform(X_aug_tr)
blind_scaled = scaler.transform(blind)
```

## Gradient Boosting Tree
The implementation of the Gradient Boosting Tree (GBT) takes a number of important parameters. We will be using cross-validation to select the best values for the following parameters used in the XGBoost classifier:

"max depth": Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
"n_estimators": the number of the boosting trees.
"learning_rate": It is also called "eta". Step size shrinkage used in update to prevents overfitting.
"min_child_weight": Minimum sum of instance weight (hessian) needed in a child.
"subsample": Subsample ratio of the training instances.
"colsample_bytree": Subsample ratio of columns when constructing each tree. More details about the parameters of XGBoost can be found at: https://xgboost.readthedocs.io/en/latest/parameter.html#
The parameter tuning was conducted using python package hyperopt.
The hyperopt packages uses the parameter search algorithm based on the Bayesian theory. The previously used parameter $\Theta$1 will create a expected post-loss function F, and the new parameter $\Theta$2 is derived to maximum the post-loss function. The $\Theta$2 is used to fit the data and if the score is better, $\Theta$2 would be used to update post-loss function F, and $\Theta$3 will be derived from maximizing the new F, and the process goes on until the number of iteration reached.

```python
space={
    'max_depth': hp.quniform("max_depth", 1, 5, 1),
    'n_estimators': hp.quniform("n_estimators", 100, 1100,200),
    'learning_rate': hp.quniform("learning_rate", 0.05,0.2,0.03),
    'min_child_weight': hp.quniform("min_child_weight", 1,15,5),
    'subsample': hp.quniform("subsample", 0.7,1,0.1),
    'colsample_bytree':hp.quniform("colsample_bytree", 0.7,1,0.1)    
    }


def objective(space):
    clf = xgb.XGBClassifier(max_depth = int(space['max_depth']),
                            n_estimators = int(space['n_estimators']),
                           learning_rate = float(space['learning_rate']),
                            min_child_weight = int(space['min_child_weight']),
                            subsample = float(space['subsample']),
                            colsample_bytree = float(space['colsample_bytree']),
                           tree_method='gpu_hist', gpu_id=0, nthread=-1)
    eval_set = [(X_test,y_test)]
    clf.fit(X_train, y_train,eval_set=eval_set,
        eval_metric="mlogloss", early_stopping_rounds=30,verbose=True)

    pred = clf.predict(X_test)
    cv_conf = confusion_matrix(y_test, pred)

    print('Optimized facies classification accuracy = %.2f' % accuracy(cv_conf))

return{'loss':1-accuracy(cv_conf), 'status': STATUS_OK }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=300,
            trials=trials)

print(best)        
```

# Model Validation with Blind Data
To evaluate the accuracy of our classifier, we predicted the lithofacies label using the well we left out and compared to the actual ones. The predictor features had been pre-processed the same way as the training data set. Our gradient boosting trees based classifier achieved an accuracy of 0.80 on the training data set, which is better than the commonly used SVM model. On the blind test data, the accuracy is 0.56, so there is room for experiment. If we count misclassification with adjacent faces as correct, the classifier has an accuracy close to 0.87.
```python
predicted_labels = clf_hp.predict(blind_scaled)
conf = confusion_matrix(blind_y, predicted_labels)
print('Facies classification accuracy = %f' % accuracy(conf))
print('Adjacent facies classification accuracy = %f' % accuracy_adjacent(conf, adjacent_facies))
```
*Facies classification accuracy = 0.559406
Adjacent facies classification accuracy = 0.866337*

We compared the predicted and labeled facies on the depth profiles:
![alt]({{ site.url }}{{ site.baseurl }}/images/04_facies/04_profile2.png)

# Summary
The 9 well-log data set was used to develop a lithology facies classification model. There are 5 well-log measurements and 2 geologically derived features. We expanded the predictor feature numbers to 435, using augment, gradient, and polynomial terms. We tried two classification algorithms, SVM and Gradient Boosting Trees. The performance of gradient boosting trees was better than SVM. The classifier achieved an accuracy of 0.80 on the training data set and 0.56 on the blind left-out set. There training dataset is still relatively small, and the imbalanced data may be the reason that the model performance dropped sharply in the test data set.
