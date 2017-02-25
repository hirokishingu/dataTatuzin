import math
import numpy as np
import pandas as pd
import operator
from scipy.misc import comb
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from itertools import product

# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import cross_val_score
    from sklearn.cross_validation import GridSearchCV
else:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV



print(50 * "=")
print("Section: Learning with ensembles")
print(50 * "-")

def ensemble_error(n_classifier, error):
  k_start = math.ceil(n_classifier / 2.0)
  probs = [comb(n_classifier, k) * error**k * (1 - error)**(n_classifier - k)
            for k in range(k_start, n_classifier + 1)]
  return sum(probs)

print("Ensemble error", ensemble_error(n_classifier=11, error=0.25))

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error)
              for error in error_range]

plt.plot(error_range,
          ens_errors,
          label="Ensemble error",
          linewidth=2)

plt.plot(error_range,
          error_range,
          linestyle="--",
          label="Base error",
          linewidth=2)

plt.xlabel("Base error")
plt.ylabel("Base/Ensemble error")
plt.legend(loc="upper left")
plt.grid()

plt.show()

print(50 * '=')
print('Section: Implementing a simple majority vote classifier')
print(50 * '-')

np.argmax(np.bincount([0, 0, 1],
                      weights=[0.2, 0.2, 0.6]))
ex = np.array([[0.9, 0.1],
              [0.8, 0.2],
              [0.4, 0.6]])

p = np.average(ex,
              axis=0,
              weights=[0.2, 0.2, 0.6])

print("Averaged prediction", p)
print("np.argmax(p):", np.argmax(p))

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, classifiers, vote="classlabel", weights=None):
    self.classifiers = classifiers
    self.named_classifiers = {key: value for key, value
                              in _name_estimators(classifiers)}
    self.vote = vote
    self.weights = weights

  def fit(self, X, y):
    if self.vote not in ("probability", "classlabel"):
      raise ValueError("vote must be 'probability' or ' classlabel'"
                      "; got (vote=%r)"
                      % self.vote)

    if self.weights and len(self.weights) != len(self.classifiers):
      raise ValueError("Number of classifiers and weights must be equal"
                        "; got %d weights, %d classifiers"
                        % (len(self.weights), len(self.classifiers)))

    self.lablenc_ = LabelEncoder()
    self.lablenc_.fit(y)
    self.classes_ = self.lablenc_.classes_
    self.classifiers_ = []
    for clf in self.classifiers:
      fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
      self.classifiers_.append(fitted_clf)
    return self

  def predict(self, X):
    if self.vote == "probability":
      maj_vote = np.argmax(self.predict_proba(X), axis=1)
    else:
      predictions = np.asarray([clf.predict(X)
                                for clf in self.classifiers_]).T
      maj_vote = np.apply_along_axis(
        lambda x:
        np.argmax(np.bincount(x,
                              weights=self.weights)),
        axis=1,
        arr=predictions)
    maj_vote = self.lablenc_.inverse_transform(maj_vote)
    return maj_vote

  def predict_proba(self, X):
    probas = np.asarray([clf.predict_proba(X)
                          for clf in self.classifiers_])
    avg_proba = np.average(probas, axis=0, weights=self.weights)
    return avg_proba

  def get_params(self, deep=True):
    if not deep:
      return super(MajorityVoteClassifier, self).get_params(deep=False)
    else:
      out = self.named_classifiers.copy()
      for name, step in six.iteritems(self.named_classifiers):
        for key, value in six.iteritems(step.get_params(deep=True)):
          out["%s__%s" % (name, key)] = value
      return out

print(50 * '=')
print('Section: Combining different algorithms for'
      ' classification with majority vote')
print(50 * '-')

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test =\
  train_test_split(X, y,
                    test_size=0.5,
                    random_state=1)

clf1 = LogisticRegression(penalty="l2",
                          C=0.001,
                          random_state=0)

clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion="entropy",
                              random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric="minkowski")

pipe1 = Pipeline([["sc", StandardScaler()],
                  ["clf", clf1]])
pipe3 = Pipeline([["sc", StandardScaler()],
                  ["clf", clf3]])
clf_labels = ["Logistic Regression", "Decision Tree","Knn"]

print("10-fold cross validation:\n")
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
  scores = cross_val_score(estimator=clf,
                            X=X_train,
                            y=y_train,
                            cv=10,
                            scoring="roc_auc")
  print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels += ["Majority Voting"]
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
  scores = cross_val_score(estimator=clf,
                            X=X_train,
                            y=y_train,
                            cv=10,
                            scoring="roc_auc")
  print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))



print(50 * '=')
print('Section: Evaluating and tuning the ensemble classifier')
print(50 * '-')

colors = ["black", "orange", "blue", "green"]
linestyles = [":", "--", "-.", "-"]
for clf, label, clr, ls \
        in zip(all_clf,
               clf_labels, colors, linestyles):


  y_pred = clf.fit(X_train,
                  y_train).predict_proba(X_test)[:, 1]
  fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                  y_score=y_pred)

  roc_auc = auc(x=fpr, y=tpr)
  plt.plot(fpr, tpr,
            color=clr,
            linestyle=ls,
            label="%s (auc = %0.2f" % (label, roc_auc))

plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1],
        linestyle="--",
        color="gray",
        linewidth=2)

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

all_clf = [pipe1, clf2, pipe3, mv_clf]

x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex="col",
                        sharey="row",
                        figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        all_clf, clf_labels):
  clf.fit(X_train_std, y_train)

  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)

  axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
                                X_train_std[y_train == 0, 1],
                                c="blue",
                                marker="^",
                                s=50)

  axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                X_train_std[y_train == 1, 1],
                                c="red",
                                marker="o",
                                s=50)
  axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -4.5,
          s="Sepal width [standardized]",
          ha="center", va="center", fontsize=12)
plt.text(-10.5, 4.5,
        s="Petal length [standardized]",
        ha="center", va="center",
        fontsize=12, rotation=90)

plt.show()

print(mv_clf.get_params())

params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0]}

grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring="roc_auc")
grid.fit(X_train, y_train)

if Version(sklearn_version) < "0.18":
  for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f +/- %0.2f %r"
            % (mean_score, scores.std() / 2.0, params))

else:
  cv_keys = ("mean_test_score", "std_test_score", "params")

  for r, _ in enumerate(grid.cv_results_["mean_test_score"]):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
            grid.cv_results_[cv_keys[1]][r] / 2.0,
            grid.cv_results_[cv_keys[2]][r]))

print("Best parameters: %s" % grid.best_params_)
print("Accuracy: %.2f" % grid.best_score_)












































