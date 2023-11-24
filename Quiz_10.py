import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
column_names=['sepal-length', 'sepal-width', 'petal length', 'petal-width', 'class']
data=pd.read_csv("./data/09_irisdata.csv", names = column_names)
print(np.shape(data))
print(data.describe())
print(data.groupby('class').size())
scatter_matrix(data)
plt.savefig("./data/scatter_plot.png")
x = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

model = DecisionTreeClassifier()
kfold = KFold(n_splits=10, random_state=5, shuffle=True)
model= DecisionTreeClassifier()
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())
