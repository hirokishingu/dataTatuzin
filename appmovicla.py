from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

from vecto import vect

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                  "pkl_objects",
                  "classifier.pkl"), "rb"))
db = os.path.join(cur_dir, "reviews.sqlite")

def classify(document):
  label = {0: "negative", 1: "positive"}
  X = vect.transform([document])
  y = clf.predict(X)[0]
  proba = np.max(clf.predict_prona(X))
  return label[y], proba

def train(document, y):
  X = vect.transform([document])
  clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
  conn = sqlite3.connect(path)
  c = conn.cursou()
  c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
  conn.commit()
  conn.close()





































