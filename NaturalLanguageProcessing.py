# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the text
import re 
import nltk
# nltk.download('stopwords')  //you need this downloaed in order to run the code
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range (0, 1000):
    review = re.sub('^a-zA-Z', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0) 

# Decision Tree:
# Accuracy = 69.5%, Precision = 55.33%, Recall =  79.16%,  F1 Score = 65.13%
"""from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)"""

# Naive Bayes:
# Accuracy = 72.5%, Precision = 67.64%, Recall = 89.32%,  F1 Score = 76.98%
"""from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)"""

# Random Forest Classification:
# Accuracy = 74%, Precision = 84.93%, Recall =  60.19%,  F1 Score = 70.45%
"""from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 55, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)"""

# kernel SVM: 
# Accuracy = 48%, Precision = 0, Recall =  0,  F1 Score = 0
"""from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)"""

# SVM:
# Accuracy = 76.5%, Precision = 81.81%, Recall = 69.90%,  F1 Score = 75.49
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)
# KNN: 
# Accuracy = 64%, Precision = 78.18%, Recall = 41.74%,  F1 Score = 54.42
"""from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)"""

# Logistic Regression: 
# Accuracy = 75%, Precision = 82.71%, Recall = 65.04%,  F1 Score = 72.81
"""from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,  solver='lbfgs')
classifier.fit(x_train, y_train)"""

# Predicting the Test set and results
y_pred = classifier.predict(x_test)

# Making the Confusing Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
