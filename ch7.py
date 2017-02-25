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
















































