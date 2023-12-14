import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.preprocessing import Binarizer
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor
import sys
sys.path.append('/home/cjw/cmlcompiler/python')
from cmlcompiler.algorithms.linear import logistic_regression
from cmlcompiler.model import build_model
import time

X = np.random.rand(1000, 100)
y = np.random.randint(2, size=1000)
batch_size = 1000
# 预处理算法
clf = Binarizer()
# 线性模型
# clf = LogisticRegression()
# clf = LinearRegression()
# clf = SGDClassifier()
# 树形模型
# clf = DecisionTreeClassifier()
# clf = RandomForestClassifier()
# clf = ExtraTreeRegressor()
# clf = ExtraTreesClassifier()
# 支持向量机
# clf = LinearSVR()

clf.fit(X, y)

# print(clf.score(X, y))
# print(clf.coef_)
# print(clf.intercept_)

# cmlcompiler
# clf = logistic_regression(X.shape, batch_size, False)

# print(type(clf))

load_time = []
exec_time = []
store_time = []
time_start = time.perf_counter()  # 记录开始时间

for i in range(20):
    target = "llvm"
    # 第一个参数clf为sklearn_model
    model = build_model(clf, X.shape, batch_size=batch_size, target=target)
    out = model.run(X, True)
    load_time.append(out[0])
    # print(out[0])
    exec_time.append(out[1])
    store_time.append(out[2])

time_end = time.perf_counter()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print("The load_time:", np.mean(load_time))
print("The exec_time:", np.mean(exec_time))
print("The store_time:", np.mean(store_time))
print("The total_time:", time_sum)
