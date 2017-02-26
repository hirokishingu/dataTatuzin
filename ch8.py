import pyprind
import pandas as pd
import os
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import GridSearchCV
else:
    from sklearn.model_selection import GridSearchCV

#############################################################################
print(50 * '=')
print('Section: Obtaining the IMDb movie review dataset')
print(50 * '-')

print('!! This script assumes that the movie dataset is located in the'
      ' current directory under ./aclImdb')

_ = input('Please hit enter to continue.')

basepath = './aclImdb'

"""
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r',
                      encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']


np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)
"""



df = pd.read_csv('/Users/hrksng/Documents/python/tatuzin/movie_data.csv')
print('Excerpt of the movie dataset', df.head(3))



print(50 * '=')
print('Section: Transforming documents into feature vectors')
print(50 * '-')

count = CountVectorizer()
docs = np.array(["The sun is shining",
                  "The weather is sweet",
                  "The sun is shining and the weather is sweet"])

bag = count.fit_transform(docs)

print("Vocabulary", count.vocabulary_)
print("bag.toarray()", bag.toarray())


print(50 * '=')
print('Section: Assessing word relevancy via term frequency-inverse'
      ' document frequency')
print(50 * '-')

np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

tf_is = 2
n_docs = 3
idf_is = np.log((n_docs + 1) / (3 + 1))
tfidf_is = tf_is * (idf_is + 1)
print("tf-idf of term 'is' = %.2f" % tfidf_is)

tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
print("raw tf-idf", raw_tfidf)

l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
# l2_tfidf
print("l2 tf-idf", l2_tfidf)


print(50 * '=')
print('Section: Cleaning text data')
print(50 * '-')

print('Excerpt:\n\n', df.loc[0, 'review'][-50:])


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text


print("Preprocessor on Excerpt:\n\n", preprocessor(df.loc[0, "review"][-50:]))

res = preprocessor("</a>This :) is :( a test :-)!")
print("preprocessor on '</a>This :) is :( a test :-)!':\n\n", res)

df["review"] = df["review"].apply(preprocessor)














































