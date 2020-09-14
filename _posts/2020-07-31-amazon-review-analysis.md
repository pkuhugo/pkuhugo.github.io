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
In this project, I conducted the sentimental analysis on the kindle book reviews. In the training dataset, each book review is labeled as 'positive' or 'negative' based on the review content. The goal of this project is to build a predictive model for sentiment analysis of the new comments. I went through the basic steps of the Natural Language Processing (NLP) and used the Multinomial Naive Bayes algorithm for sentiment prediction.

# Data
I obtained the amazon book review data online and saved two important data fields for the model training. The amazon book review data include two columns. The "reviewText" is the raw user review of a certain book, and the "overall", which only has two values ("pos" and "neg") is used to represent "positive" and "negative". In total, the training dataset has 126,871 reviews. For the text data, I'll perform the following steps to extract features for the model training.
1. Tokenization<br>
Tokenization splits a phrase, sentence, paragraph, or an entire text document into smaller units, such as characters, word ,or subword (n-gram characters). In this tokenized form, we can count the number of words in the text, or count the frequency of the word. It's the start of extracting useful information from text data. In this project, I used the NLTK python package for tokenization.<br>
2. Normalization<br>
The first step of normalization is to remove the stopwords and punctuation. Stopwords are the most common words used in any natural language. For the purpose of analyzing test data and extract useful information, these commonly used words may not carry much value. The most common words used in a text are "the", "is", "in" etc. The NLTK package includes a collection of common stopwords and would be used to remove them. Punctuations sometimes have useful information, such as emotion and parsing of the sentence. In this project, I was dealing with large corpora, thus I removed the punctuations for simplicity.
After removing the stopwords and punctuation, I applied lemmatization to the remaining tokens of word. In English, words appear in several forms. For example, the verb "to walk" may appear as "take", "takes" or "taken". The base form, "take", that one might look up in a dictionary, is called the lemma for the word. Stemming is the process of reducing the inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the language. For example, the root of "takes", "take" and "taken" is "tak".<br>
Lemmatization is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words that have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster. The reduced "accuracy" may not matter for some applications. In fact, when used within information retrieval systems, stemming improves query recall accuracy, or true positive rate, when compared to lemmatization. Nonetheless, stemming reduces precision, or true negative rate, for such systems. For instance, the word "better" has "good" as its lemma. This link is missed by stemming, as it requires a dictionary look-up.
3. bag of words<br>
The normalized tokens would be saved in a field, like a bag of words. The bag-of-words includes information about the vocabulary of the text and the frequency of known words. The information about the order or structure of words in the document is lost using the bag-of-word model.

```python
def preprocessing(line):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    tokens = []
    line = str(line).translate(translation) #replace puncutuation
    line = nltk.word_tokenize(line.lower()) #tokenize

    for t, tag in nltk.pos_tag(line):
        #remove stopwords
        if t not in stop:
            stemmed = lemmatizer.lemmatize(t,tag_map[tag[0]])
            tokens.append(stemmed)
    return ' '.join(tokens)

#preprocess for all
start = time.time()
amz_df['bow'] = [preprocessing(p) for p in amz_df['reviewText']]
end = time.time()
print(end - start)  
```
4. tf-idf vectorizer
Tf: the term frequency of a word in a document. There are several ways of calculating this frequency, with the simplest being a raw count of instance a word appears in a document.
idf: inverse document-frequency
Tf-idf = $$tf(t,d) \times idf(t)$$<br>
<br>
$$
idf(t) = log{\frac{1 + nd}{1 + df(d, t)}} + 1
$$
<br>

where n is the total number of documents in the document set and df(t) is the document frequency of t; the document frequency is the number of documents in the document set that contain the term t. The effect of adding “1” to the idf in the equation above is that terms with zero idf, i.e., terms that occur in all documents in a training set, will not be entirely ignored.<br>

![](http://www.onemathematicalcat.org/Math/Algebra_II_obj/Graphics/log_base_gt1.gif)

Tf-idf works by increasing proportionally to the number of times a word appears in a document but is offset by the number of documents that contain the word. So, words that are common in every document, such as this, what and I, ranked low even though their count is large since they don't mean much to the particular document. However, if the word "amazing" appears many times in a review text, while not appearing many times in others, it probably means that it's relevant and the reviewer liked it.

```python
#Since CountVectorizer and TfidfTransfomer are often used together
tf_vec = TfidfVectorizer(vocabulary=topwords)

# Extract features from training set
# Vocabulary is from topwords
start = time.time()
train_features  =  tf_vec.fit_transform(X_train)
end = time.time()
print(end-start)  
```

# Model Training and evaluation
## [Multinomial NB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

The multinomial Naive Bayes classifier is suitable for **classification with discrete features** (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
```python
from sklearn.naive_bayes import MultinomialNB
mnb_model = MultinomialNB()

# Train Model
start = time.time()
mnb_model.fit(train_features, y_train)
end = time.time()
print("Multinomial NB model trained in %f seconds" % (end-start))

# Predict
pred = mnb_model.predict(test_features)
print(pred)

# Metrics
print(metrics.classification_report(y_true=y_test, y_pred=pred))
```

I build a simple sentiment analysis model using the bag-of-word and Multinomial Naive Bayes Model, the prediction accuracy is about 0.81. If I train the model using N-gram, N=2 in feature extraction, the accuracy can be improved to 0.83. In the end, the model can be saved and used for future review sentiment predictions. I wrote a small prediction function for prediction.

```python
# Predict a new sentence
# vectorizer needs to be pre-fitted
# At the end of the project, the function signature should be something like:
# predict_new(sentent: str, vec, model) -> str

def predict_new(sentence):
    sentence = preprocessing(sentence)
    features = tf_vec.transform([sentence])
    pred = mnb_model.predict(features)
    return pred[0]

predict_new("I can't stop reading it")    
```
And the result is 'pos'!

# Summary
I built a sentiment analysis model based on the amazon book review data. The bag-of-word method and Multinomial Naive Bayes algorithm are used for model training. For NLP analysis, the preprocessing on the text data is very important for feature extraction. In this project, I went through tokenization, removing stopwords and punctuation, lemmatization ,and tf-idf. In the end, the model was able to identify positive and negative reviews.
