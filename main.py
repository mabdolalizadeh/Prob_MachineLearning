import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from regressions import Regressions


matplotlib.use('TkAgg')

dataset = pd.read_csv("sleep-dataset.csv")

# cleaning dataset
dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset.select_dtypes(include=['number'])

cov_matrix = dataset.cov()  # covariance
correlation_matrix = dataset.corr(method='pearson')  # correlation

# plotting
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

regression = Regressions(dataset, correlation_matrix)

# plot prediction
plt.scatter(regression.y_test, regression.y_pred, color='blue', label='Prediction')
plt.plot([min(regression.y_test), max(regression.y_test)],
         [min(regression.y_test), max(regression.y_test)], 'r--', label='Perfect fit')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.show()