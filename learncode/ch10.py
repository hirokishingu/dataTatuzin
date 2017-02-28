import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

print(50 * '=')
print('Section: Exploring the Housing dataset')
print(50 * '-')

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'housing/housing.data',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print('Dataset excerpt:\n\n', df.head())


print(50 * "=")
print("Section: Visualizing the important characteristics of a dataset")
print(50 * "-")

sns.set(style="whitegrid", context="notebook")
cols = ["LSTAT", "INDUS", "NOX", "RM", "MEDV"]

sns.pairplot(df[cols], size=2.5)

plt.show()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                  cbar=True,
                  annot=True,
                  square=True,
                  fmt=".2f",
                  annot_kws={"size": 15},
                  yticklabels=cols,
                  xticklabels=cols)

plt.show()
sns.reset_orig()


print(50 * '=')
print('Section: Solving regression for regression'
      ' parameters with gradient descent')
print(50 * '-')

class LinearRegressionGD(object):

  def __init__(self, eta=0.001, n_iter=20):
    self.eta = eta
    self.n_iter = n_iter

  def fit(self, X, y):
    self.w_ = np.zeros(1 + X.shape[1])
    self.cost_ = []

    for i in range(self.n_iter):
      output = self.net_input(X)
      errors = (y - output)
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      cost = (errors**2).sum() / 2.0
      self.cost_.append(cost)
    return self

  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]

  def predict(self, X):
    return self.net_input(X)

X = df[["RM"]].values
y = df["MEDV"].values

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.show()

def lin_regplot(X, y, model):
  plt.scatter(X, y, c="lightblue")
  plt.plot(X, model.predict(X), color="red", linewidth=2)
  return

lin_regplot(X_std, y_std, lr)
plt.xlabel("Average number of rooms [RM] (standardized)")
plt.ylabel("Price in $1000\'s [MEDV] (standardized)")
plt.show()

print("Slope: %.3f" % lr.w_[1])
print("Intercept: %.3f" % lr.w_[0])

num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))


print(50 * '=')
print('Section: Estimating the coefficient of a'
      ' regression model via scikit-learn')
print(50 * '-')


slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print("Slope: %.3f" % slr.coef_[0])
print("Intercept: %.3f" % slr.intercept_)

lin_regplot(X, y, slr)
plt.xlabel("Average nimber of rooms [RM]")
plt.ylabel("Price in $1000\'s [MEDV]")

plt.show()

Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print("Slope: %.3f" % w[1])
print("Intercept: %.3f" % w[0])




print(50 * '=')
print('Section: Fitting a robust regression model using RANSAC')
print(50 * '-')


if Version(sklearn_version) < '0.18':
    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100,
                             min_samples=50,
                             residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                             residual_threshold=5.0,
                             random_state=0)
else:
    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100,
                             min_samples=50,
                             loss='absolute_loss',
                             residual_threshold=5.0,
                             random_state=0)

ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c="blue", marker="o", label="Inliers")
plt.scatter(X[outlier_mask], y[outlier_mask],
            c="lightgreen", marker="s", label="Outliers")
plt.plot(line_X, line_y_ransac, color="red")
plt.xlabel("Average number of rooms [RM]")
plt.ylabel("Price in $1000\'s [MEDV]")
plt.legend(loc="upper left")

plt.show()

print("Slope: %.3f" % ransac.estimator_.coef_[0])
print("Intercept: %.3f" % ransac.estimator_.intercept_)




















































