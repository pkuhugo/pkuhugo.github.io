---
title: "Predict Default Loans with Lending Club Data"
date: 2020-07-07
tags: [data wrangling,  machine learning]
header:
  image: "/images/01_lendingclub/loan_by_year2.jpg"
excerpt: "Data Wrangling, Machine Learning"
mathjax: "true"
---

# Introduction
LendingClub is the world's largest peer-to-peer lending platform. The company enables borrowers to create personal loans between $1,000 and $40,000. The investors can search and browse the loan listings on LendingClub website and select loans that they want to invest based on the information supplied about the borrower. Investor make money from interest. LendingClub makes money by charging borrowers an origination fee and investor a service fee.

This project is to use the historical loan data from LendingClub to build a model that can predict the loan status (whether or not the loan will be default or fully paid). The model also finds the most important factors which impact the prediction from the user information provided by the borrowers. The Lending Club investors can use these results to make better-informed decisions on note choice.

# Data description
In this project, I mainly worked on LendingClub historical data from 2007 to 2018 on Kaggle. The dataset is 1.55 GB of Size including 2260701 loans and each of which has 150 associated features.

On the other hand, the historical data include some features that are not immediately available when the borrowers create the personal loan. Thus, I used the latest data of the recently listed loans through LendingClub API. Only the features included in the listed loans would be used for model prediction, and other features in the historical data will be excluded. I will match the feature field between the two datasets.

Match the data feature field:
```python
# For the recent data, I will change the upper case characters to lower characters. For the historical data, I'll remove the underscore '_' from the feature names.
app_df = app_df.rename(columns = lambda x:x.lower())
hst_df = hst_df.rename(columns = lambda x: x.replace("_", ""))

#find out the different fields and match them
hst_diff = set(hst_df.columns) - set(app_df.columns)
app_diff = set(app_df.columns) - set(hst_df.columns)

#only the features existing in both historical and list loan data would be Used
same_col = set(hst_df.columns)&set(app_df.columns)
hst_match_df = hst_df[same_col]
```
After feature field matching, we excluded the features not avaiable in the recent listed loan data. There are 93 features in the historical data that can be used for model building.

# Feature Selection
Here's a [full description](http://rstudio-pubs-static.s3.amazonaws.com/290261_676d9bb194ae4c9882f599e7c0a808f2.html) of each features.

After feature field matching, there are 93 features left. But not all of the features would contribute to the default loan prediction. I conducted the following steps to exclude the features without much information.
1. I'll drop the features with 50% of the data missing
2. Split the data in to 'Fully paid' and 'Charged off' group, and use hypothesis test to compare the distributions between the two groups. If the distribution of the feature is not statistically significant different, it will be dropped
3. Remove the highly correlated features

After the feature selection, I inspected the features one by one in order to get some intuitive understanding of the data. Here are some examples of the statistical examination of the key features.

*Interest Rate* <br>
![alt]({{ site.url }}{{ site.baseurl }}/images/01_lendingclub/interest_rate.JPG)

*FICO Score*<br>
![alt]({{ site.url }}{{ site.baseurl }}/images/01_lendingclub/fico.JPG)

*Grade*<br>
![alt]({{ site.url }}{{ site.baseurl }}/images/01_lendingclub/grade.JPG)

*Home ownership*<br>
![alt]({{ site.url }}{{ site.baseurl }}/images/01_lendingclub/home.JPG)

*Income verification*<br>
![alt]({{ site.url }}{{ site.baseurl }}/images/01_lendingclub/verify.JPG)

In total, 23 features were dropped due to more than 50% data missing. I also dropped *accnowdelinq*, *delinqamnt*, *chargeoffwithin12mths* from the feature fields because of the large P values in the hypothesis test. *numsats*, *tothicredlim*, *installment*, *fundedamnt*, *ficorangehigh*, *numrevtlbalgt0* are also dropped because they are highly correlated to some other features. *zipcode* is dropped now because it has too many category values. It may be added to the feature field if it can be related to other information, such as the average income. *emptitle* is also dropped, because it is a text feature which might need Natural Language Processing for it to add value to the prediction.

# Feature Engineering
The categorical feature needs to be converted to the numerical feature in order to use them in the model training. If the categorical values have rank information (e.g., *grade* and *subgrade*), they can be converted through label encoding. If there's no ordering amongst the values of the categorical feature (e.g., *purpose* and *homeownership*), the on-hot-encoding is utilized in the numerical conversion.
Label encoding:
```python
#grade and subgrade use label encoding because the rank matters
grade_dic = {"A":1,
            "B":2,
            "C":3,
            "D":4,
            "E":5,
            "F":6,
            "G":7}
hst_drop_df.grade = hst_drop_df.grade.map(grade_dic)
hst_drop_df.subgrade = hst_drop_df.subgrade.apply(lambda x:(grade_dic[x[0]]-1)*5+int(x[1]))
```
One-hot encoding:
```python
#For the categorical data without ordering
dummy_list = ['addrstate', 'applicationtype', 'emplength',  'homeownership', 'initialliststatus', 'purpose',  'term', 'verificationstatus']
for f in dummy_list:
    dummy_df = pd.get_dummies(hst_drop_df[f], prefix = f)
    hst_drop_df = hst_drop_df.drop([f], axis=1)
    hst_drop_df = pd.concat((hst_drop_df, dummy_df), axis = 1)
```
# Model Training and Evaluation
The early 80% percent of the historical data were used for the model training, and the later 20% data were used for model evaluation.
Train/Test split:
```python
#I'll split the train/test sample at 8:2
df_train = hst_drop_df.loc[hst_drop_df['issued']<hst_drop_df['issued'].quantile(0.8)]
df_test = hst_drop_df.loc[hst_drop_df['issued']>hst_drop_df['issued'].quantile(0.8)]
```
The performance metrics of the 3 models (i.e., logistic regression, random forest, and gradient boosting tree (GBT)) are shown in Figure 1. In the training data sets, GBT has the highest AUC score. However, the GBT score is lowest among the three indicating that the model is overfitting. The highest AUC score is from the random forest model. The feature importance analysis showed that the most important factors are the interest rate, loan term length, applicant's credit health, besides the grade assigned by Lending Club.

*Model evaluation: ROC Curves*<br>
![alt]({{ site.url }}{{ site.baseurl }}/images/01_lendingclub/model.jpg)

*Feature importance*<br>
![alt]({{ site.url }}{{ site.baseurl }}/images/01_lendingclub/features.jpg)

# Conclusion
I conducted a quick EDA and prediction model building to evaluate the potential to use lending club historical data to predict the loan default risk. Based on my result, the interest rate, loan term length, applicant's credit score and account numbers were the most important factors when evaluating the loan default risk. The investors can leverage the information available when the loan is created online to quickly identify the good asset to invest. However, the accuracy of this current model has a great potential to improve. More information could be added to the training data set by relating the *zipcode* to meaningful features (e.g. average income). Also, carefully fine-tuned model can also improve the model performance. There are still a lot potentials to improve the result of this project.




## H2 Heading

### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$
