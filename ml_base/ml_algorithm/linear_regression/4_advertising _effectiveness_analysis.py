import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    # 标准化
from sklearn.linear_model import LinearRegression, SGDRegressor # 方程解析法, 随机梯度下降
from sklearn.metrics import mean_squared_error  # 均方差MSE

# 加载数据集
dataset = pd.read_csv("../../../data/advertising.csv")

# 数据处理
dataset.dropna(inplace=True)
dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset.info()
print(dataset.head())


# 划分数据集
X = dataset.drop("Sales", axis=1)
y = dataset["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征工程(只有数值型, 量纲不同)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 两种方法求解模型
# 1. 解析法
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)
print(f"lr 斜率: {model_lr.coef_}")
print(f"lr 截距: {model_lr.intercept_}")
print(f"lr 均方差: {mean_squared_error(y_test, pred_lr)}")
# 2. SGD
model_sgd = SGDRegressor()
model_sgd.fit(X_train, y_train)
pred_sgd = model_sgd.predict(X_test)
print(f"sgd 斜率: {model_sgd.coef_}")
print(f"sgd 截距: {model_sgd.intercept_}")
print(f"sgd 均方差: {mean_squared_error(y_test, pred_sgd)}")


