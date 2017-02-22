import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < "0.18":
  from sklearn.grid_search import train_test_split
else:
  from sklearn.model_selection import train_test_split



print(50 * "=")
print("Section: First steps with scikit-learn")
print(50 * "-")

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print("Class labels:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(50 * "=")
print("Section: Training a perceptron via scikit-learn")
print(50 * "-")

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
print("Y array shape", y_test.shape)

y_pred = ppn.predict(X_test_std)
print("Misclassified samples: %d" % (y_test != y_pred).sum())
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))


def versiontuple(v):
  return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
  markers = ("s", "x", "o", "^", "v")
  colors = ("red", "blue", "lightgreen", "gray", "cyan")
  cmap = ListedColormap(colors[:len(np.unique(y))])

  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))

  Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())

  #クラスごとにサンプルをプロット
  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                alpha=0.8, c=cmap(idx),
                marker=markers[idx], label=cl)
  #テストサンプるを目立たせる（点を○で表示）
  if test_idx:
    if not versiontuple(np.__version__) >= versiontuple("1.9.0"):
      X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
      warnings.warn("Please update to Numpy 1.9.0 or newer")
    else:
      X_test, y_test = X[test_idx, :], y[test_idx]

    plt.scatter(X_test[:,0],
                X_test[:, 1],
                c="",
                alpha=1.0,
                linewidths=1,
                marker="o",
                s=55, label="test set")

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")

plt.show()

print(50 * "=")
print("Section: Logistic regression intuition and conditional probabilities")
print(50 * "-")

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color="k")
plt.ylim(-0.1, 1.1)
plt.xlabel("z")
plt.ylabel("$\phi (z)$")

plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()


print(50 * "=")
print("Section: Learning the weights of the logistic cost function")
print(50 * "=")

def cost_1(z):
  return - np.log(sigmoid(z))

def cost_0(z):
  return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label="J(w) if y=1")

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle="--", label="J(w) if y=0")

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel("$\phi$(z)")
plt.ylabel("J(w)")
plt.legend(loc="best")

plt.show()


print(50 * "=")
print("Section: Training a logistic regression model with scikit-learn")
print(50 * "-")

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")

plt.show()

print("Predicted probabilities", lr.predict_proba(X_test_std[0, :].reshape(1, -1)))


print(50 * "=")
print("Section: Trackling overfitting via regularization")
print(50 * "-")

weights, params = [], []
for c in np.arange(-5, 5):
  lr = LogisticRegression(C=10**c, random_state=0)
  lr.fit(X_train_std, y_train)
  weights.append(lr.coef_[1])
  params.append(10**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0],
          label="petal length")
plt.plot(params, weights[:, 1], linestyle="--",
          label="petal width")
plt.ylabel("weigt coefficient")
plt.xlabel("c")
plt.legend(loc="upper left")
plt.xscale("log")

plt.show()



print(50 * "=")
print("Section: Dealing with the nonlinearly"
        "separable case using slach variables")
print(50 * "-")

svm = SVC(kernel="linear", C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.show()


print(50 * "=")
print("Section: Solving non-linear problems using a kernel SVM")
print(50 * "-")

#乱数種を指定
np.random.seed(0)
#標準正規分布に従う乱数で２００行２列の行列を生成
X_xor = np.random.randn(200, 2)
#２つの引数に対して排他的論理和を実行
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                      X_xor[:, 1] > 0)
#排他的論理和の値が真の場合は１、偽の場合は−１を割り当てる
y_xor = np.where(y_xor, 1, -1)

#ラベル１を青のxでプロット
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c="b", marker="x",
            label="1")
#ラベル−１を赤の四角でプロット
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c="r",
            marker="s",
            label="-1")
#軸の範囲を指定
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc="best")

plt.show()



print(50 * "=")
print("Section: Using the kernel trick to find separating hyperplanes"
        "inhigher dimensional space")
print(50 * "-")

svm = SVC(kernel="rbf", random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                        classifier=svm)
plt.legend(loc="upper left")
plt.show()

svm = SVC(kernel="rbf", random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")

plt.show()

svm = SVC(kernel="rbf", random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")

plt.show()



print(50 * "=")
print("Section: Decision tree learning")
print(50 * "-")

#ジニ不順度の関数を定義
def gini(p):
  return p * (1 - p) + (1 - p) * (1 - (1 - p))

#エントロピーの関数を定義
def entropy(p):
  return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

#分類誤差の関数を定義
def error(p):
  return 1 - np.max([p, 1 - p])
#確率を表す配列を生成（０から0.99まで0.01刻み）
x = np.arange(0.0, 1.0, 0.01)

#配列の値を元にエントロピー、分類誤差を計算
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
#エントロピー（２種）、ジニ不順度、分類誤差のそれぞれをループ処理
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ["Entropy", "Entropy (scaled)",
                            "Gini Impurity", "Misclassification Error"],
                            ["-", "-", "--", "-."],
                            ["black", "lightgray", "red", "green", "cyan"]):
  line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

#凡例の設定（中央の上に配置）
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)
#２本の水平の破線を引く
ax.axhline(y=0.5, linewidth=1, color="k", linestyle="--")
ax.axhline(y=1.0, linewidth=1, color="k", linestyle="--")
plt.ylim([0, 1.1])
plt.xlabel("p(i=1)")
plt.ylabel("Impurity Index")

plt.show()




print(50 * "=")
print("Section: Building a decision tree")
print(50 * "-")
#エオントロピーを指標とする決定木のインスタンスを生成
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
#決定木のモデルにトレーニングデータを適合させる
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel("petal length [cm]")
plt.ylabel("petal width [cm]")
plt.legend(loc="upper left")

plt.show()




print(50 * "=")
print("Section : Combining weak to strong learners via random forests")
print(50 * "-")

# エントロピーを指標とするランダムフォレストのインスタンスを生成
forest = RandomForestClassifier(criterion="entropy",
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)
#ランダムフォレストのモデルにトレーニングデータを適合させる
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel("petal length [cm]")
plt.ylabel("petal width [cm]")
plt.legend(loc="upper left")
plt.show()





























