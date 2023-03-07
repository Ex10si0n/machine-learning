import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/abalone/abalone.data"
)

abalone = pd.read_csv(url, header=None)

abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

abalone = abalone.drop("Sex", axis=1)
# abalone["Rings"].hist(bins=15)
# plt.show()

correlation_matrix = abalone.corr()
# print(correlation_matrix["Rings"].sort_values(ascending=False))
"""
    Rings             1.000000
    Shell weight      0.627574
    Diameter          0.574660
    Height            0.557467
    Length            0.556720
    Whole weight      0.540390
    Viscera weight    0.503819
    Shucked weight    0.420884
    Name: Rings, dtype: float64
"""

# KNN applies on k=3
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values

k = 3

new_data_point = np.array([
    0.569552,
    0.446407,
    0.154437,
    1.016849,
    0.439051,
    0.222526,
    0.291208,
])

distances = np.linalg.norm(X - new_data_point, axis=1)
nearest_neighbor_indices = distances.argsort()[:k]
nearest_neighbor_rings = y[nearest_neighbor_indices]

prediction = nearest_neighbor_rings.mean()
class_neighbors = np.array(["A", "B", "B", "C"])
stats.mode(class_neighbors)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

knn_model = KNeighborsRegressor(n_neighbors=3)

knn_model.fit(X_train, y_train)

train_predictions = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_predictions)
rmse = sqrt(mse)

test_predictions = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_predictions)
rmse = sqrt(mse)

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=test_predictions, s=50, cmap=cmap
)
f.colorbar(points)
plt.show()
