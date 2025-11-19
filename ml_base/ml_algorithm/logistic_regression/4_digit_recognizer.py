import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 加载数据集
dataset = pd.read_csv("../../../data/train.csv")
dataset.dropna(inplace=True)

# 划分数据集
X = dataset.drop("label", axis=1)
y = dataset["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 特征工程, 归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义模型
model = LogisticRegression(max_iter=1000)

# 模型训练
model.fit(X_train, y_train)

# 模型评估
score = model.score(X_test, y_test)
print(f"预测准确率: {score:.4f}")

# 预测某个新图像表示的数字
digit = X_test[123, :].reshape(1, -1)
digit_pred = model.predict(digit)
print(f"预测结果: {digit_pred}")
print(y_test.iloc[123])
print(model.predict_proba(digit))
plt.imshow(digit.reshape(28, 28), cmap="gray")
plt.show()