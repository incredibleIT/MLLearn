import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression   # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures    # 构建多项式特征完成多项式回归
from sklearn.model_selection import train_test_split    # 划分训练集和测试集
from sklearn.metrics import mean_squared_error  # 均方误差损失函数


# plt.rcParams['font.sans-serif'] = ['SiHei']
# plt.rcParams['axes.unicode_minus'] = False


"""
1. 生成数据
2. 划分训练集和测试集(验证集)
3. 定义模型(线性回归模型)
4. 训练模型
5. 预测结果, 计算损失
"""

# 1. 生成数据, 基于sinx [-1, 1]
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, X.shape).reshape(-1, 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(X, y, c='y')
ax[1].scatter(X, y, c='y')
ax[2].scatter(X, y, c='y')

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# (1) 欠拟合数据
X_train_1 = X_train
X_test_1 = X_test
# (2) 刚好拟合数据
poly = PolynomialFeatures(degree=5)     # 用来构建5次多项式特征
X_train_2 = poly.fit_transform(X_train)
X_test_2 = poly.transform(X_test)
# (3) 过拟合数据
poly_over = PolynomialFeatures(degree=20)
X_train_3 = poly_over.fit_transform(X_train)
X_test_3 = poly_over.transform(X_test)


# 3. 定义模型
model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()

# 4. 训练模型
model1.fit(X_train_1, y_train)
model2.fit(X_train_2, y_train)
model3.fit(X_train_3, y_train)

# 5. 预测结果, 计算损失
y_pred_1 = model1.predict(X_test_1)
test_loss_1 = mean_squared_error(y_test, y_pred_1)
train_loss_1 = mean_squared_error(y_train, model1.predict(X_train_1))

y_pred_2 = model2.predict(X_test_2)
test_loss_2 = mean_squared_error(y_test, y_pred_2)
train_loss_2 = mean_squared_error(y_train, model2.predict(X_train_2))

y_pred_3 = model3.predict(X_test_3)
test_loss_3 = mean_squared_error(y_test, y_pred_3)
train_loss_3 = mean_squared_error(y_train, model3.predict(X_train_3))

titles = ['Underfitting', 'Good Fit', 'Overfitting']
for i in range(3):
    ax[i].set_title(titles[i])
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
# 画出第一种欠拟合结果
ax[0].plot(X, model1.predict(X), c='b')
ax[0].text(-3, 1, f'test_loss_1: {test_loss_1:.4f}')
ax[0].text(-3, 0.8, f'train_loss_1: {train_loss_1:.4f}')
# 画出第二种刚好拟合结果
ax[1].plot(X, model2.predict(poly.fit_transform(X)), c='b')
ax[1].text(-3, 1, f'test_loss_2: {test_loss_2:.4f}')
ax[1].text(-3, 0.8, f'train_loss_2: {train_loss_2:.4f}')

# 画出第三种过拟合结果
ax[2].plot(X, model3.predict(poly_over.fit_transform(X)), c='b')
ax[2].text(-3, 1, f'test_loss_3: {test_loss_3:.4f}')
ax[2].text(-3, 0.8, f'train_loss_3: {train_loss_3:.4f}')

plt.show()
