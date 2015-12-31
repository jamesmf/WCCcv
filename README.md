# WCCcv

# What's Cooking Challenge:

This repo explores two approaches to the Kaggle task of identifying cuisine type from recipe data.  First, it treats recipes as documents and learns multiple LDA representations of documents, then learns using that feature space. Second, it adds a cuisine keyword to each recipe (e.g. "IRISH_RECIPE") and learns a word2vec representation of the data. From this we learn an embedding of the output space (whose first two principal components are plotted below) which we can use for as our targets.

![image](https://cloud.githubusercontent.com/assets/7809188/12065439/2343b872-afa5-11e5-9e09-a7b798893711.jpg)
