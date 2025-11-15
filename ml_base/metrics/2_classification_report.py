"""
机器学习方法解决分类问题
"""
from sklearn.datasets import make_classification        # 构建分类数据集
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.linear_model import LogisticRegression     # 逻辑回归模型
from sklearn.metrics import classification_report, roc_auc_score  # 分类评估报告

# 1. 构建数据集
X, y = make_classification(n_samples=1000, n_features=40, n_classes=2, random_state=42)

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义机器学习模型
model = LogisticRegression()

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测结果
y_pred = model.predict(X_test)

# 6. 生成评估报告
report = classification_report(y_test, y_pred)
print(report)

# 获取模型预测样本是正类的概率
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(auc_score)

