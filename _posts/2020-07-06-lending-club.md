---
title: "Predict Default Loans with Lending Club Data"
date: 2020-07-07
tags: [data wrangling,  machine learning]
header:
  image: "/images/perceptron/percept.jpg"
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
After feature field matching, there are 93 features left. But not all of the features would contribute to the default loan prediction. I conducted the following steps to exclude the features without much information.
1. I'll drop the features with 50% of the data missing
2. Split the data in to 'Fully paid' and 'Charged off' group, and use hypothesis test to compare the distributions between the two groups. If the distribution of the feature is not statistically significant different, it will be dropped
3. Remove the highly correlated features

After the feature selection, I inspected the features one by one. Here are some examples of the statistical examination of the key features.

*Interest Rate*
![alt]({{ site.url }}{{ site.baseurl }}/images/01_lendingclub/interest_rate.JPG)



# Feature Engineering





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
