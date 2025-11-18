import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer   # 列转换器
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 1. 加载数据集
heart_disease_data = pd.read_csv('../../../data/heart_disease.csv')

# 数据清洗, 处理缺失值
heart_disease_data.dropna(inplace=True)
# 2. 划分数据集
X = heart_disease_data.drop("是否患有心脏病", axis=1)  # 特征
y = heart_disease_data["是否患有心脏病"]   # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 特征工程
# 数值型特征
numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]
# 类别型特征
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]
# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("bin", "passthrough", binary_features)
    ]
)
# 进行特征转换
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# 简单实现
# 4. 创建模型
# knn = KNeighborsClassifier(n_neighbors=3)

# 5. 训练模型
# knn.fit(X_train, y_train)

# 6. 模型评估计算预测准确率
# score = knn.score(X_test, y_test)
# print(f"预测准确率: {score:.4f}")
#
# 7. 保存模型
# joblib.dump(value=knn, filename="./knn_model")

# 8. 加载模型对新数据进行预测
# knn_loaded = joblib.load("./knn_model")
# y_pred = knn_loaded.predict(X_test[10:11])
# print(f"预测结果: {y_pred}, 真实值: {y_test[10]}")

# 交叉验证选择模型
# 9. 模型评估与超参数调优
knn = KNeighborsClassifier()
# 定义网格搜索参数列表
param_grid = {"n_neighbors": list(range(1, 11)), "weights": ["uniform", "distance"]}
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10)

# 模型训练
grid_search.fit(X_train, y_train)

# 评估结果
results = pd.DataFrame(grid_search.cv_results_).to_string()
print(results)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳准确率: {grid_search.best_score_:.4f}")
print(f"最佳模型: {grid_search.best_estimator_}")

# 使用最佳模型进行测试评估
knn = grid_search.best_estimator_
print(f"预测准确率: {knn.score(X_test, y_test):.4f}")