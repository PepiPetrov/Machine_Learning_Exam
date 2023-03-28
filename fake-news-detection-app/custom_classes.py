import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer

import re

from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import softmax


from sklearn.svm import LinearSVC


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    Does lemmatization and stopwords removal.
    """
    def __init__(self):
        self.stopwords = stopwords.words("english")
    
    def normalize(self, document):
        lemma = WordNetLemmatizer()
        stemmed_content = re.sub('[^a-zA-Z]',' ', document)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.strip()
        stemmed_content = stemmed_content.split()
        stemmed_content = [lemma.lemmatize(word) for word in stemmed_content if not word in self.stopwords]
        stemmed_content = ' '.join(stemmed_content)

        return stemmed_content

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        result = []
        for document in documents:
            result.append(self.normalize(document))
        
        return result

class LinearSVCwithProbabilities(LinearSVC):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return softmax(d_2d)