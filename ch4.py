import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from itertools import combinations
import matplotlib.pyplot as plt

from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < "0.18":
  from sklearn.grid_search import train_test_split
else:
  from sklearn.model_selection import train_test_split



print(50 * "=")
print("Section: Dealing with missing data")
print(50 * "-")

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)
print("\n\nExecuting df.isnull().sum():")
print(df.isnull().sum())









































