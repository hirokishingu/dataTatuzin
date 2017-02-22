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


print(50 * "=")
print("Section: Eliminating samples or features with missing values")
print(50 * "-")

print("\n\nExecuting df.dropna()")
#欠測値を含む行を削除
print(df.dropna())

print("\n\nExecuting df.dropna()")
#欠測値を含む列を削除
print(df.dropna(axis=1))

print("\n\nExecuting df.dropna(thresh=4)")
print("(drop rows that have not at least 4 non-NaN values)")
#非NaN値が４つ未満の行を削除
print(df.dropna(thresh=4))

print("\n\nExecuting df.dropna(how='all')")
print("(only drop rows where all columns are NaN")
#全ての列がNaNである行だけを削除する
print(df.dropna(how="all"))

print("\n\nExecuting df.dropna(subset=['C'])")
print("(only drop rows where NaN appear in specific columns (here: 'C'))")
#特定の列（この場合は”C”）に NaNNあNが含まれている行だけを削除
print(df.dropna(subset=["C"]))



print(50 * "=")
print("Section: Imuputing missing velues")
print(50 * "-")
#欠測値補完のインスタンスを生成（平均値補完）
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
#データを適合tekigou
imr = imr.fit(df)
#補完を実行
imputed_data = imr.transform(df.values)

print("Input Array:\n", df.values)
print("Imputed Data:\n", imputed_data)







































