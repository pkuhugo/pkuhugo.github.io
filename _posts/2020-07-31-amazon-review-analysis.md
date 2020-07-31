---
title: "Sentiment Analysis with Amazon Book Review Data"
date: 2020-07-31
tags: [machine learning, NLP]
header:
  image: "/images/03_amazon/words.jpg"
excerpt: "Machine Learning, NLP"
mathjax: "true"
---
# Introduction
In this project, I conducted the sentimental analysis on the kindle book reviews. In the training dataset, each book review is labeled as 'positive' or 'negative' based on the review content. The goal of this project is to build a predict model to analysis sentiment of the new comments. I'll go through the basic steps of the Natural Language Processing (NLP) and use the Multinomial Naive Bayes algorithm for sentiment prediction.

# Data
The amazon book review data include two columns. The "reviewText" is the raw user review of a certain book, and the "overall", which only has two values ("pos" and "neg") is used to represent "positive" and "negative". In total, the training dataset has 126,871 reviews. For the text data, I'll follow the steps to extract features for the model training.
1. tokenize
2. normalization
3. bag of words
4. tf-idf vectorizer

## lemmatization and stemming difference:
In many languages, words appear in several inflected forms. For example, in English, the verb 'to walk' may appear as 'walk', 'walked', 'walks' or 'walking'. The base form, 'walk', that one might look up in a dictionary, is called the lemma for the word. The association of the base form with a part of speech is often called a lexeme of the word.

Lemmatisation is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster. The reduced "accuracy" may not matter for some applications. In fact, when used within information retrieval systems, stemming improves query recall accuracy, or true positive rate, when compared to lemmatisation. Nonetheless, stemming reduces precision, or true negative rate, for such systems.[5]

For instance:

The word "better" has "good" as its lemma. This link is missed by stemming, as it requires a dictionary look-up.
The word "walk" is the base form for the word "walking", and hence this is matched in both stemming and lemmatisation.
The word "meeting" can be either the base form of a noun or a form of a verb ("to meet") depending on the context; e.g., "in our last meeting" or "We are meeting again tomorrow". Unlike stemming, lemmatisation attempts to select the correct lemma depending on the context.

## tf-idf  term weighting
- Tf: term-frequency
- idf: inverse document-frequency
- Tf-idf = $tf(t,d) \times idf(t)$

$$
idf(t) = log{\frac{1 + nd}{1 + df(d, t)}} + 1
$$

![](http://www.onemathematicalcat.org/Math/Algebra_II_obj/Graphics/log_base_gt1.gif)

# Model Training and evaluation
## [Multinomial NB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

The multinomial Naive Bayes classifier is suitable for **classification with discrete features** (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
